package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.InvalidUserInputException;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.LineFit;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions.FCSFit;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Pair;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Range;
import ij.IJ;

import java.awt.*;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The {@code DiffusionLawModel} class represents the data model for diffusion law analysis.
 * It handles data preparation, fitting, and calculation of diffusion law parameters.
 */
public class DiffusionLawModel {
    private static final int MAX_POINTS = 30;
    private static final int DIMENSION_ROI = 7;
    private static final int MINIMUM_POINTS = 4;
    private final ExpSettingsModel interfaceSettings;
    private final FitModel interfaceFitModel;
    private final BleachCorrectionModel interfaceBleachCorrectionModel;
    private final ImageModel imageModel;
    private final Runnable resetCallback;
    private String mode = "All";
    private int binningStart = 1;
    private int binningEnd = 5;
    private int calculatedBinningStart = -1;
    private int calculatedBinningEnd = -1;
    private int fitStart = 1;
    private int fitEnd = 5;
    private double[] effectiveArea;
    private double[] time;
    private double[] standardDeviation;
    private double minValueDiffusionLaw = Double.MAX_VALUE;
    private double maxValueDiffusionLaw = -Double.MAX_VALUE;
    private Map<Double, double[][]> psfResults = null;
    private double intercept = -1;
    private double slope = -1;

    /**
     * Initializes the model with the provided experimental settings, image data, and fitting model.
     *
     * @param settings              Experimental settings model.
     * @param imageModel            Image model containing the data.
     * @param fitModel              Fitting model used for the correlation data.
     * @param bleachCorrectionModel BleachCorrection model used for applying bleach correction.
     * @param resetCallback         Callback to handle resetting results.
     */
    public DiffusionLawModel(ExpSettingsModel settings, ImageModel imageModel, FitModel fitModel,
                             BleachCorrectionModel bleachCorrectionModel, Runnable resetCallback) {
        this.interfaceSettings = settings;
        this.interfaceFitModel = fitModel;
        this.interfaceBleachCorrectionModel = bleachCorrectionModel;

        this.imageModel = imageModel;
        this.resetCallback = resetCallback;
    }

    /**
     * Fits a pixel at the specified coordinates and updates the diffusion coefficient statistics
     * for the current binning setting.
     *
     * @param settings   the experimental settings model used for fitting.
     * @param fitModel   the fitting model containing parameters for fitting the pixel data.
     * @param correlator the correlator used to compute correlation data for the pixel.
     * @param averageD   Array for average diffusion coefficients.
     * @param varianceD  Array for variances.
     * @param pixelModel the model representing the pixel's data and fit parameters.
     * @param x          the x-coordinate of the pixel to be fitted.
     * @param y          the y-coordinate of the pixel to be fitted.
     * @param index      the index corresponding to the current binning setting.
     * @return 1 if the pixel was successfully fitted, 0 otherwise.
     */
    private int fitPixelAndAddD(ExpSettingsModel settings, FitModel fitModel, Correlator correlator, double[] averageD,
                                double[] varianceD, PixelModel pixelModel, int x, int y, int index) {
        try {
            correlator.correlatePixelModel(pixelModel, imageModel.getImage(), x, y, x, y, settings.getFirstFrame(),
                    settings.getLastFrame());
            fitModel.standardFit(pixelModel, Constants.ITIR_FCS_2D, correlator.getLagTimes());
        } catch (Exception e) {
            IJ.log(String.format("Fail to correlate points for x=%d, y=%d with error: %s", x, y, e.getMessage()));
            return 0;
        }

        if (pixelModel.isFitted()) {
            averageD[index] += pixelModel.getFitParams().getD();
            varianceD[index] += Math.pow(pixelModel.getFitParams().getD(), 2);

            // Return 1 to count this element in the total elements.
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * Calculates the diffusion law for the current binning range.
     */
    public void calculateDiffusionLaw() {
        resetResults();

        // update the binning calculated, this is used to know if a calculation was run.
        calculatedBinningStart = binningStart;
        calculatedBinningEnd = binningEnd;

        // create a new settings model to be able to update the binning size separately from the interface.
        ExpSettingsModel settings = new ExpSettingsModel(this.interfaceSettings);
        // set the CCF to 0 for the diffusion law analysis
        settings.setCCF(new Dimension(0, 0));

        Pair<double[], double[]> results = calculateDiffusionLaw(binningStart, binningEnd, settings, true);
        double[] averageD = results.getLeft();
        double[] varianceD = results.getRight();
        computeDiffusionLawParameters(averageD, varianceD);
    }

    /**
     * Calculates diffusion law parameters across a binning range.
     *
     * @param binningStart Start of binning range.
     * @param binningEnd   End of binning range.
     * @param settings     Experimental settings model.
     * @param progress     Whether to show progress in ImageJ.
     * @return Pair containing average diffusion coefficients and their variances.
     */
    public Pair<double[], double[]> calculateDiffusionLaw(int binningStart, int binningEnd, ExpSettingsModel settings,
                                                          boolean progress) {
        // init a new fit model based on the interface parameters and fix the parameters
        FitModel fitModel = new FitModel(settings, interfaceFitModel);
        fitModel.setFix(true);

        BleachCorrectionModel bleachCorrectionModel =
                new BleachCorrectionModel(settings, interfaceBleachCorrectionModel);

        Correlator correlator = new Correlator(settings, bleachCorrectionModel, fitModel);
        PixelModel pixelModel = new PixelModel();

        effectiveArea = new double[binningEnd - binningStart + 1];
        double[] averageD = new double[binningEnd - binningStart + 1];
        double[] varianceD = new double[binningEnd - binningStart + 1];

        for (int currentBinning = binningStart; currentBinning <= binningEnd; currentBinning++) {
            settings.setBinning(new Point(currentBinning, currentBinning));
            settings.updateSettings();

            Range[] ranges = settings.getAllArea(imageModel.getDimension());
            Range xRange = ranges[0];
            Range yRange = ranges[1];

            int index = currentBinning - binningStart;

            effectiveArea[index] =
                    FCSFit.getFitObservationVolume(settings.getParamAx(), settings.getParamAy(), settings.getParamW()) *
                            Constants.DIFFUSION_COEFFICIENT_BASE;

            int numElements = xRange.stream()
                    .mapToInt(x -> yRange.stream()
                            .mapToInt(y -> fitPixelAndAddD(settings, fitModel, correlator, averageD, varianceD,
                                    pixelModel, x, y, index))
                            .sum())
                    .sum();

            // Do the average and convert to um^2/s
            averageD[index] *= Constants.DIFFUSION_COEFFICIENT_BASE / numElements;
            // normalize the average of the square
            varianceD[index] *= Math.pow(Constants.DIFFUSION_COEFFICIENT_BASE, 2) / numElements;
            varianceD[index] = varianceD[index] - Math.pow(averageD[index], 2);

            // show the progress
            if (progress) {
                IJ.showProgress(index, binningEnd - binningStart + 1);
            }
        }

        return new Pair<>(averageD, varianceD);
    }

    /**
     * Computes diffusion law parameters.
     *
     * @param averageD  Array of average diffusion coefficients.
     * @param varianceD Array of variances.
     */
    private void computeDiffusionLawParameters(double[] averageD, double[] varianceD) {
        time = new double[effectiveArea.length];
        standardDeviation = new double[effectiveArea.length];

        minValueDiffusionLaw = Double.MAX_VALUE;
        maxValueDiffusionLaw = -Double.MAX_VALUE;

        for (int i = 0; i < effectiveArea.length; i++) {
            time[i] = effectiveArea[i] / averageD[i];
            standardDeviation[i] = effectiveArea[i] / Math.pow(averageD[i], 2) * Math.sqrt(varianceD[i]);

            minValueDiffusionLaw = Math.min(minValueDiffusionLaw, averageD[i] - varianceD[i]);
            maxValueDiffusionLaw = Math.max(maxValueDiffusionLaw, averageD[i] + varianceD[i]);
        }
    }

    /**
     * Determines the Point Spread Function (PSF) by iterating over a range of values.
     *
     * @param start        Starting PSF value.
     * @param end          Ending PSF value.
     * @param step         Step size for PSF values.
     * @param binningStart Start of binning range.
     * @param binningEnd   End of binning range.
     * @return A pair containing the minimum and maximum values of the diffusion law.
     */
    public Pair<Double, Double> determinePSF(double start, double end, double step, int binningStart, int binningEnd) {
        // Validate input ranges
        if (binningStart <= 0 || binningEnd <= binningStart || step <= 0 || start >= end) {
            throw new IllegalArgumentException("Invalid PSF or binning range.");
        }

        ExpSettingsModel settings = new ExpSettingsModel(this.interfaceSettings);

        psfResults = new LinkedHashMap<>();

        double minValue = Double.MAX_VALUE;
        double maxValue = -Double.MAX_VALUE;

        int currentStep = 0;
        int numSteps = (int) Math.ceil((end - start) / step) + 1;

        for (double currentPSF = start; currentPSF <= end; currentPSF += step) {
            settings.setSigma(String.valueOf(currentPSF));
            settings.updateSettings();

            Pair<double[], double[]> diffLawResults = calculateDiffusionLaw(binningStart, binningEnd, settings, false);
            double[] averageD = diffLawResults.getLeft();
            double[] varianceD = diffLawResults.getRight();

            double[][] results = new double[3][binningEnd - binningStart + 1];
            for (int currentBinning = binningStart; currentBinning <= binningEnd; currentBinning++) {
                int index = currentBinning - binningStart;
                results[0][index] = currentBinning;
                results[1][index] = averageD[index];

                // error bars: SEM of diffusion coefficient
                results[2][index] = varianceD[index] / Math.sqrt(
                        (double) imageModel.getWidth() / currentBinning * (double) imageModel.getHeight() /
                                currentBinning);

                minValue = Math.min(minValue, results[1][index] - results[2][index]);
                maxValue = Math.max(maxValue, results[1][index] + results[2][index]);
            }

            psfResults.put(currentPSF, results);

            IJ.showProgress(currentStep++, numSteps);
        }

        // reset the results to delete diffusion law parameters
        resetResults();
        return new Pair<>(minValue, maxValue);
    }


    /**
     * Returns a subarray corresponding to the current fitting range.
     *
     * @param array The source array.
     * @return The subarray within the fitting range.
     */
    private double[] getFitSegment(double[] array) {
        return Arrays.copyOfRange(array, fitStart - calculatedBinningStart, fitEnd - calculatedBinningStart + 1);
    }

    /**
     * Performs a linear fit on the calculated diffusion law data.
     *
     * @return The fitted line data.
     */
    public double[][] fit() {
        if (effectiveArea == null) {
            throw new RuntimeException("Please run the diffusion law calculation before");
        } else if (fitStart < calculatedBinningStart || fitEnd > calculatedBinningEnd) {
            throw new RuntimeException("Fit start/end not are out of ranges");
        }

        double[] segmentEffectiveArea = getFitSegment(effectiveArea);
        double[] segmentTime = getFitSegment(time);
        double[] segmentStandardDeviation = getFitSegment(standardDeviation);

        LineFit lineFit = new LineFit();
        double[] result =
                lineFit.doFit(segmentEffectiveArea, segmentTime, segmentStandardDeviation, fitEnd - fitStart + 1);

        intercept = result[0];
        slope = result[1];

        double[][] fitFunction = new double[2][segmentEffectiveArea.length];
        fitFunction[0] = segmentEffectiveArea;

        for (int i = 0; i < segmentEffectiveArea.length; i++) {
            fitFunction[1][i] = intercept + slope * segmentEffectiveArea[i];
        }

        return fitFunction;
    }

    /**
     * Resets the binning and fitting range values to their default states.
     * This method is typically called when the user switches to ROI (Region of Interest) mode.
     *
     * @param reset if true, resets the range values and sets the mode to "ROI"; otherwise, sets the mode to "All".
     * @return the updated mode string, either "ROI" or "All".
     */
    public String resetRangeValues(boolean reset) {
        if (reset) {
            this.binningEnd = 5;
            this.fitStart = 1;
            this.fitEnd = 5;

            this.mode = "ROI";
        } else {
            this.mode = "All";
        }

        return this.mode;
    }

    /**
     * Resets the calculated results of the diffusion law analysis.
     * <p>
     * This method clears the stored data and resets the calculated binning ranges to their initial states.
     * It is typically called when the user changes the analysis parameters or when a new calculation is initiated.
     */
    public void resetResults() {
        this.calculatedBinningStart = -1;
        this.calculatedBinningEnd = -1;

        minValueDiffusionLaw = Double.MAX_VALUE;
        maxValueDiffusionLaw = -Double.MAX_VALUE;

        effectiveArea = null;
        time = null;
        standardDeviation = null;

        intercept = -1;
        slope = -1;
    }

    /**
     * Set the default range for binning and fit based on the image model dimension.
     */
    public void setDefaultRange() {
        resetResults();

        binningStart = 1;
        fitStart = 1;

        int size = Math.min(imageModel.getWidth(), imageModel.getHeight());
        if (interfaceSettings.isOverlap()) {
            binningEnd = size + 1 - MINIMUM_POINTS;
        } else {
            binningEnd = size / MINIMUM_POINTS;
        }

        fitEnd = binningEnd;
    }

    // Getter and setter methods with input validation for binning and fitting ranges.

    public int getBinningStart() {
        return binningStart;
    }

    public void setBinningStart(String binningStart) {
        int start = Integer.parseInt(binningStart);
        if (start <= 0 || start >= this.binningEnd) {
            throw new InvalidUserInputException("Binning start out of range.");
        } else if (calculatedBinningStart != -1 && calculatedBinningStart != start) {
            resetCallback.run();
        }
        this.binningStart = start;
    }

    public int getBinningEnd() {
        return binningEnd;
    }

    public void setBinningEnd(String binningEnd) {
        int end = Integer.parseInt(binningEnd);
        if (end >= MAX_POINTS || end <= this.binningStart) {
            throw new InvalidUserInputException("Binning end out of range");
        } else if (calculatedBinningEnd != -1 && calculatedBinningEnd != end) {
            resetCallback.run();
        }
        this.binningEnd = end;
    }

    public int getFitStart() {
        return fitStart;
    }

    public void setFitStart(String fitStart) {
        int start = Integer.parseInt(fitStart);
        if (start <= 0 || start >= this.fitEnd || start < this.binningStart || start > this.binningEnd) {
            throw new InvalidUserInputException("Fit start out of range.");
        }
        this.fitStart = start;
    }

    public int getFitEnd() {
        return fitEnd;
    }

    public void setFitEnd(String fitEnd) {
        int end = Integer.parseInt(fitEnd);
        if (end <= this.fitStart || end > this.binningEnd) {
            throw new InvalidUserInputException("Fit end out of range.");
        }
        this.fitEnd = end;
    }

    public int getDimensionRoi() {
        return DIMENSION_ROI;
    }

    public String getMode() {
        return mode;
    }

    public double[] getEffectiveArea() {
        return effectiveArea;
    }

    public double[] getTime() {
        return time;
    }

    public double[] getStandardDeviation() {
        return standardDeviation;
    }

    public double getIntercept() {
        return intercept;
    }

    public double getSlope() {
        return slope;
    }

    public double getMinValueDiffusionLaw() {
        return minValueDiffusionLaw;
    }

    public double getMaxValueDiffusionLaw() {
        return maxValueDiffusionLaw;
    }

    public Map<Double, double[][]> getPsfResults() {
        return psfResults;
    }
}
