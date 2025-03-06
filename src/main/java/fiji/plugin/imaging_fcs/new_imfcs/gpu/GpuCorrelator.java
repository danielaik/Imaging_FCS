package fiji.plugin.imaging_fcs.new_imfcs.gpu;

import java.util.stream.IntStream;

import fiji.plugin.imaging_fcs.gpufitImFCS.GpufitImFCS;
import fiji.plugin.imaging_fcs.new_imfcs.model.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Range;
import ij.IJ;

/**
 * Handles GPU-accelerated correlation and fitting operations for imaging FCS data.
 * This class coordinates data preparation, correlation calculations, and optional fitting
 * using the GPU for high-performance processing.
 */
public class GpuCorrelator {
    private final ExpSettingsModel settings;
    private final GpuParameters gpuParameters;
    private final Correlator correlator;
    private final ImageModel imageModel;
    private final FitModel fitModel;

    /**
     * Constructs a new GPU correlator with all required components.
     *
     * @param settings The experimental settings model
     * @param bleachCorrectionModel Model containing bleach correction parameters
     * @param imageModel Model representing the image data
     * @param fitModel Model containing fitting parameters and functions
     * @param isNBcalculation Flag indicating if number and brightness calculation is needed
     * @param correlator Correlator for managing correlation data.
     */
    public GpuCorrelator(ExpSettingsModel settings, BleachCorrectionModel bleachCorrectionModel, ImageModel imageModel,
            FitModel fitModel, boolean isNBcalculation, Correlator correlator, Range xRange, Range yRange) {
        this.settings = settings;
        this.correlator = correlator;
        this.imageModel = imageModel;
        this.gpuParameters = new GpuParameters(settings, bleachCorrectionModel, imageModel, fitModel, isNBcalculation,
                correlator, xRange, yRange);
        this.fitModel = fitModel;
    }

    /**
     * Performs correlation and optional fitting on a range of pixels.
     *
     * @param xRange The range of x coordinates to process
     * @param yRange The range of y coordinates to process
     * @param fit Whether to perform fitting after correlation
     */
    public void correlateAndFit(Range xRange, Range yRange, boolean fit) {
        // Define the number of steps. If we fit, we have the correlation, the fit, and then the plot.
        int numberOfStep = fit ? 3 : 2;

        float[] pixels = gpuParameters.getIntensityData(xRange, yRange, false);

        double[] bleachCorrectionParams = gpuParameters.calculateBleachCorrectionParams(pixels);
        // gpuParameters.applyBleachCorrection(pixels, bleachCorrectionParams);

        double[] nbMean = new double[gpuParameters.width * gpuParameters.height];
        double[] nbCovariance = new double[gpuParameters.width * gpuParameters.height];

        int arraySize = gpuParameters.width * gpuParameters.height * gpuParameters.chanum;

        double[] pixels1 = new double[arraySize];
        double[] blockVarianceArray = new double[arraySize];
        double[] blocked1D = new double[arraySize];

        GpufitImFCS.calcACF(pixels, pixels1, blockVarianceArray, nbMean, nbCovariance, blocked1D,
                bleachCorrectionParams, IntStream.of(correlator.getSampleTimes()).asDoubleStream().toArray(),
                correlator.getLags(), gpuParameters);

        IJ.showProgress(1, numberOfStep);

        PixelModel[][] pixelModels = createPixelModels(pixels1, blockVarianceArray, blocked1D, xRange.getStart(), yRange.getStart());

        if (fit) {
            // Perform GPU fit
            GpuFitter fitter = new GpuFitter(gpuParameters, fitModel, settings, correlator);
            fitter.fit(pixelModels, pixels1, blockVarianceArray, xRange.getStart(), yRange.getStart());

            IJ.showProgress(2, numberOfStep);
        }
    }

    /**
     * Creates pixel models from correlation results.
     *
     * @param pixels1 Array containing correlation function values
     * @param blockVarianceArray Array containing variance data
     * @param blocked1D Array containing blocked data
     * @return 2D array of pixel models with correlation data
     */
    private PixelModel[][] createPixelModels(double[] pixels1, double[] blockVarianceArray, double[] blocked1D, int roiStartX, int roiStartY) {
        PixelModel[][] pixelModels = correlator.getPixelModels();

        if (pixelModels == null) {
            pixelModels = new PixelModel[imageModel.getWidth()][imageModel.getHeight()];
        }

        for (int x = 0; x < gpuParameters.width; x++) {
            for (int y = 0; y < gpuParameters.height; y++) {
                PixelModel pixelModel = new PixelModel();
                // Allocate arrays to hold per-channel data.
                double[] correlationFunction = new double[gpuParameters.chanum];
                double[] varianceCF = new double[gpuParameters.chanum];
                double[] standardDeviationCF = new double[gpuParameters.chanum];


                for (int channel = 0; channel < gpuParameters.chanum; channel++) {
                    int index = channel * gpuParameters.width * gpuParameters.height + y * gpuParameters.width + x;
                    correlationFunction[channel] = pixels1[index];
                    varianceCF[channel] = 1.0 / blockVarianceArray[index];
                    standardDeviationCF[channel] = Math.sqrt(varianceCF[channel]);
                }

                pixelModel.setCorrelationFunction(correlationFunction);
                pixelModel.setVarianceCF(varianceCF);
                pixelModel.setStandardDeviationCF(standardDeviationCF);
                pixelModel.setBlocked((int) blocked1D[x + gpuParameters.width + gpuParameters.width * gpuParameters.height]);

                pixelModels[(x + roiStartX) * gpuParameters.pixbinX][(y + roiStartY) * gpuParameters.pixbinY] = pixelModel;
            }
        }

        correlator.setPixelModels(pixelModels);

        return pixelModels;
    }

    // public static native void calcACF(float[] pixels, double[] pixels1, double[] blockvararray,
    //         double[] NBmeanGPU, double[] NBcovarianceGPU,
    //         double[] blocked1D, double[] bleachcorr_params,
    //         double[] samp, int[] lag, GpuParameters ACFInputParams);
}
