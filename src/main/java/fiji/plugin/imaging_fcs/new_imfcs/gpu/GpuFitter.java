package fiji.plugin.imaging_fcs.new_imfcs.gpu;

import fiji.plugin.imaging_fcs.gpufit.*;
import fiji.plugin.imaging_fcs.new_imfcs.enums.FitFunctions;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions.FCSFit;

/**
 * Handles GPU-accelerated fitting of correlation functions.
 * This class prepares data for fitting, performs parallel fits on GPU,
 * and processes the results for each pixel in the image.
 */
public class GpuFitter {
    private final ExpSettingsModel settings;
    private final GpuParameters gpuParams;
    private final fiji.plugin.imaging_fcs.new_imfcs.model.FitModel fitModel;
    private final Correlator correlator;
    // private final float[] filterArray;

    /**
     * Constructs a new GPU fitter with all required components.
     *
     * @param gpuParams  GPU parameters for the fitting operation
     * @param fitModel   Model containing fitting parameters and functions
     * @param settings   The experimental settings model
     * @param correlator The correlator implementation providing lag times
     */
    public GpuFitter(GpuParameters gpuParams, fiji.plugin.imaging_fcs.new_imfcs.model.FitModel fitModel,
                     ExpSettingsModel settings, Correlator correlator) {
        this.settings = settings;
        this.gpuParams = gpuParams;
        this.fitModel = fitModel;
        this.correlator = correlator;
    }

    /**
     * Determines the appropriate model for fitting based on settings.
     *
     * @param selectedModel The model name from settings
     * @return The corresponding GPU fit model
     */
    private static Model determineModel(FitFunctions selectedModel) {
        if (selectedModel == FitFunctions.SPIM_FCS_3D) {
            return Model.ACF_NUMERICAL_3D;
        } else {
            return Model.ACF_1D;
        }
    }

    /**
     * Calculates the residuals of a correlation function given the fitted and original arrays.
     *
     * @param fittedCF            the fitted correlation function values
     * @param correlationFunction the original correlation function values
     * @return a new array containing the residuals
     */
    private static double[] calculateResiduals(double[] fittedCF, double[] correlationFunction) {
        double[] residuals = new double[fittedCF.length];
        for (int i = 0; i < fittedCF.length; i++) {
            residuals[i] = correlationFunction[i] - fittedCF[i];
        }
        return residuals;
    }

    /**
     * Performs parallel fitting on multiple pixels' correlation functions using GPU.
     *
     * @param pixelModels        2D array of pixel models to fit
     * @param pixels1            Array containing correlation function values
     * @param blockVarianceArray Array containing variance data
     */
    public void fit(PixelModel[][] pixelModels, double[] pixels1, double[] blockVarianceArray, int roiStartX,
                    int roiStartY) {
        Model model = determineModel(settings.getFitModel());
        int numberFits = gpuParams.width * gpuParams.height;
        int numberPoints = gpuParams.fitend - gpuParams.fitstart + 1;

        float[] initialParameters = prepareInitialParameters(model, numberFits);
        float[] userInfo = prepareUserInfo(numberPoints);
        float[] data = prepareData(pixels1, numberFits, numberPoints);
        float[] weights = prepareWeights(blockVarianceArray, numberFits, numberPoints);
        Boolean[] parametersToFit = prepareParametersToFit(model);

        GpuFitModel gpufitModel = new GpuFitModel(numberFits, numberPoints, true, model, GpuFitModel.TOLERANCE,
                GpuFitModel.FIT_MAX_ITERATIONS, gpuParams.bleachcorr_order, parametersToFit, Estimator.LSE,
                numberPoints * Float.BYTES);

        gpufitModel.data.put(data);
        gpufitModel.weights.put(weights);
        gpufitModel.initialParameters.put(initialParameters);
        gpufitModel.userInfo.put(userInfo);

        FitResult fitResult = Gpufit.fit(gpufitModel);

        processFitResults(pixelModels, model, numberFits, fitResult, roiStartX, roiStartY);
        // applyIntensityFilter(pixelModels);
    }

    /**
     * Prepares the initial parameter array for multiple fits based on the model.
     * This method retrieves the parameter values from the {@code fitModel} and
     * GPU-related parameters, then replicates them across all fits.
     *
     * @param model      The model defining the number of parameters per fit.
     * @param numberFits The number of fits to perform.
     * @return A float array containing the initial parameters for all fits.
     */
    private float[] prepareInitialParameters(Model model, int numberFits) {
        // Retrieve float parameter values from FitModel
        float[] modelParameters = fitModel.getFloatParameterValues();

        // Additional GPU parameters
        float[] gpuParameters = new float[]{
                (float) settings.getParamAx(),
                (float) settings.getParamAy(),
                (float) settings.getParamW(),
                (float) settings.getParamZ(),
                (float) settings.getParamRx(),
                (float) settings.getParamRy(),
                (float) FCSFit.getFitObservationVolume(settings.getParamAx(), settings.getParamAy(),
                        settings.getParamW()),
                (float) fitModel.getQ2(),
                (float) fitModel.getQ3(),
                (float) settings.getEmLambdaInterface(),
                (float) settings.getNA()
        };

        // Combine model and GPU parameters
        float[] trueParameters = new float[modelParameters.length + gpuParameters.length];
        System.arraycopy(modelParameters, 0, trueParameters, 0, modelParameters.length);
        System.arraycopy(gpuParameters, 0, trueParameters, modelParameters.length, gpuParameters.length);

        // Create the final array for all fits
        float[] initialParameters = new float[numberFits * model.numberParameters];
        for (int i = 0; i < numberFits; i++) {
            System.arraycopy(trueParameters, 0, initialParameters, i * model.numberParameters, model.numberParameters);
        }

        return initialParameters;
    }

    /**
     * Prepares the user info array containing lag times for fitting.
     *
     * @param numberPoints Number of data points in each fit
     * @return Array of lag times for the fitting range
     */
    private float[] prepareUserInfo(int numberPoints) {
        float[] userInfo = new float[numberPoints];
        double[] lagTimes = correlator.getLagTimes();
        for (int i = 0; i < numberPoints; i++) {
            userInfo[i] = (float) lagTimes[i + gpuParams.fitstart];
        }
        return userInfo;
    }

    /**
     * Prepares correlation function data for fitting.
     *
     * @param pixels1      Array containing correlation function values
     * @param numberFits   Number of fits to perform
     * @param numberPoints Number of data points in each fit
     * @return Formatted data array for GPU fitting
     */
    private float[] prepareData(double[] pixels1, int numberFits, int numberPoints) {
        float[] data = new float[numberFits * numberPoints];
        int counter = 0;
        for (int y = 0; y < gpuParams.height; y++) {
            for (int x = 0; x < gpuParams.width; x++) {
                for (int z = gpuParams.fitstart; z <= gpuParams.fitend; z++) {
                    int index = z * gpuParams.width * gpuParams.height + y * gpuParams.width + x;
                    data[counter++] = (float) pixels1[index];
                }
            }
        }
        return data;
    }

    /**
     * Prepares weights array for weighted fitting.
     *
     * @param blockVarianceArray Array containing variance data
     * @param numberFits         Number of fits to perform
     * @param numberPoints       Number of data points in each fit
     * @return Weights array for GPU fitting
     */
    private float[] prepareWeights(double[] blockVarianceArray, int numberFits, int numberPoints) {
        float[] weights = new float[numberFits * numberPoints];
        int counter = 0;
        for (int y = 0; y < gpuParams.height; y++) {
            for (int x = 0; x < gpuParams.width; x++) {
                for (int z = gpuParams.fitstart; z <= gpuParams.fitend; z++) {
                    int index = z * gpuParams.width * gpuParams.height + y * gpuParams.width + x;
                    weights[counter++] = (float) (1.0 / blockVarianceArray[index]);
                }
            }
        }
        return weights;
    }

    /**
     * Prepares array specifying which parameters should be fitted.
     *
     * @param model The GPU fitting model
     * @return Array of Boolean flags for parameters to fit
     */
    private Boolean[] prepareParametersToFit(Model model) {
        boolean[] parametersToFit = fitModel.getParametersToFit();
        Boolean[] finalParametersToFit = new Boolean[model.numberParameters];

        int limit = Math.min(parametersToFit.length, model.numberParameters);
        for (int i = 0; i < limit; i++) {
            finalParametersToFit[i] = parametersToFit[i]; // Auto-boxing boolean to Boolean
        }

        for (int i = limit; i < model.numberParameters; i++) {
            finalParametersToFit[i] = false;
        }

        return finalParametersToFit;
    }

    /**
     * Processes and stores fit results in pixel models.
     *
     * @param pixelModels 2D array of pixel models to update
     * @param model       The GPU fitting model used
     * @param numberFits  Number of fits performed
     * @param fitResult   Results from GPU fitting
     */
    private void processFitResults(PixelModel[][] pixelModels, Model model, int numberFits, FitResult fitResult,
                                   int roiStartX, int roiStartY) {
        int numFreeParams = (int) fitModel.getNonHeldParameterValues().length;
        int numParameters = PixelModel.FitParameters.NUM_PARAMETERS;

        for (int x = 0; x < gpuParams.width; x++) {
            for (int y = 0; y < gpuParams.height; y++) {
                PixelModel pixel =
                        pixelModels[(x + roiStartX) * gpuParams.pixbinX][(y + roiStartY) * gpuParams.pixbinY];
                if (pixel == null) {
                    continue;
                }

                int index = y * gpuParams.width + x;

                boolean converged = FitState.fromID(fitResult.states.get(index)) == FitState.CONVERGED;
                pixel.setFitted(converged);

                float chiSquare = fitResult.chiSquares.get(index);
                int degreesOfFreedom = (gpuParams.fitend - gpuParams.fitstart + 1) - numFreeParams - 1;
                pixel.setChi2(chiSquare / degreesOfFreedom);

                if (converged) {
                    double[] params = new double[numParameters];
                    for (int p = 0; p < numParameters; p++) {
                        params[p] = fitResult.parameters.get(index * model.numberParameters + p);
                    }

                    pixel.setFitParams(new PixelModel.FitParameters(params));
                    fitModel.calculateFittedCorrelationFunction(pixel, correlator.getLagTimes());
                    pixel.setResiduals(calculateResiduals(pixel.getFittedCF(), pixel.getCorrelationFunction()));
                }
            }
        }
    }
}