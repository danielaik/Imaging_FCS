package fiji.plugin.imaging_fcs.imfcs.gpu;

import fiji.plugin.imaging_fcs.gpufit.*;
import fiji.plugin.imaging_fcs.imfcs.enums.BleachCorrectionMethod;
import fiji.plugin.imaging_fcs.imfcs.model.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.imfcs.utils.Range;
import ij.IJ;
import ij.ImagePlus;

import java.awt.*;

/**
 * Holds and computes parameters for GPU-based image processing, including image dimensions,
 * binning parameters, bleach correction settings, and intensity data extraction.
 * <p>
 * The class initializes values from experimental settings, image models, and correlator parameters,
 * and provides native methods to perform binning and bleach correction on GPU as well as fallback CPU methods.
 * </p>
 */
public class GpuParameters {
    public int width; // output width
    public int height; // output height
    public int win_star; // w output x pixbinX + cfXdistance
    public int hin_star; // h output x pixbinY + cfYdistance
    public int w_temp; // win_star - pixbinX + 1
    public int h_temp; // hin_start - pixbinY + 1
    public int pixbinX;
    public int pixbinY;
    public int binningX; // binning in X axis
    public int binningY; // binning in Y axis
    public int firstframe;
    public int lastframe;
    public int framediff;
    public int cfXDistance;
    public int cfYDistance;
    public double correlatorp;
    public double correlatorq;
    public double frametime;
    public int background;
    public double mtab1; // mtab[1], used to calculate blocknumgpu.
    public double mtabchanumminus1; // mtab[chanum-1], used to calculate pnumgpu[counter_indexarray]
    public double sampchanumminus1; // samp[chanum-1], used to calculate pnumgpu[counter_indexarray]
    public int chanum;
    public boolean isNBcalculation;
    public boolean bleachcorr_gpu;
    public int bleachcorr_order;
    public int nopit;
    public int ave;
    public int fitstart;
    public int fitend;
    private ImageModel imageModel;

    /**
     * Constructs GPU parameters for camera fitting
     * TODO: This should be deleted when the camera code is refactor.
     */
    public GpuParameters() {

    }

    /**
     * Constructs GPU parameters from experimental settings, bleach correction model, image model,
     * fit model, and additional parameters.
     *
     * @param settings              Experimental settings model.
     * @param bleachCorrectionModel Bleach correction model.
     * @param imageModel            Image model containing image and background data.
     * @param fitModel              Model containing fitting parameters.
     * @param isNBcalculation       Flag for NB calculation mode.
     * @param correlator            Correlator to compute additional parameters.
     * @param xRange                Range for the X dimension.
     * @param yRange                Range for the Y dimension.
     * @throws IllegalArgumentException if calculated dimensions are invalid.
     */
    public GpuParameters(ExpSettingsModel settings, BleachCorrectionModel bleachCorrectionModel, ImageModel imageModel,
                         FitModel fitModel, boolean isNBcalculation, Correlator correlator, Range xRange,
                         Range yRange) {
        // Extract dimensions from ImageModel
        this.imageModel = imageModel;

        Point pixelBinning = settings.getPixelBinning();
        this.pixbinX = isNBcalculation ? 1 : pixelBinning.x;
        this.pixbinY = isNBcalculation ? 1 : pixelBinning.y;
        this.binningX = isNBcalculation ? 1 : settings.getBinning().x;
        this.binningY = isNBcalculation ? 1 : settings.getBinning().y;
        this.firstframe = settings.getFirstFrame();
        this.lastframe = settings.getLastFrame();
        this.framediff = this.lastframe - this.firstframe + 1;
        this.cfXDistance = isNBcalculation ? 0 : settings.getCCF().width;
        this.cfYDistance = isNBcalculation ? 0 : settings.getCCF().height;
        this.correlatorp = settings.getCorrelatorP();
        this.correlatorq = settings.getCorrelatorQ();
        this.frametime = settings.getFrameTime();

        this.nopit = bleachCorrectionModel.getNumPointsIntensityTrace();

        this.fitstart = isNBcalculation ? 0 : fitModel.getFitStart();
        this.fitend = isNBcalculation ? 0 : fitModel.getFitEnd();

        // Calculate output dimensions
        this.width = xRange.length();
        this.height = yRange.length();

        // Ensure dimensions are valid
        if (this.width <= 0 || this.height <= 0) {
            throw new IllegalArgumentException("Invalid dimensions for GPU parameters.");
        }

        // Calculate intermediate dimensions
        this.win_star = this.width * pixbinX + cfXDistance;
        this.hin_star = this.height * pixbinY + cfYDistance;
        this.w_temp = this.win_star - binningX + 1;
        this.h_temp = this.hin_star - binningY + 1;

        // Determine bleach correction
        if (settings.getBleachCorrection() == BleachCorrectionMethod.POLYNOMIAL) {
            this.bleachcorr_gpu = true;
            this.bleachcorr_order = bleachCorrectionModel.getPolynomialOrder() + 1;
        } else {
            this.bleachcorr_gpu = false;
            this.bleachcorr_order = 0;
        }

        // Average calculation
        this.ave = (int) Math.floor(this.framediff / (double) this.nopit);

        this.background = imageModel.getBackgroundModel().getConstantBackground1();

        // Example placeholders for GPU-specific parameters
        this.chanum = settings.getChannelNumber();

        correlator.calculateParameters(settings.getLastFrame() - settings.getFirstFrame());
        this.mtab1 = correlator.getNumSamples()[1];
        this.mtabchanumminus1 = correlator.getNumSamples()[this.chanum - 1];
        this.sampchanumminus1 = correlator.getSampleTimes()[this.chanum - 1];
        this.isNBcalculation = isNBcalculation;
    }

    /**
     * Native method. Find if GPU memory is sufficient to perform binning.
     *
     * @param ACFInputParams Object that encapsulates / stores the necessary
     *                       input values.
     * @return true is memory is sufficient, false otherwise.
     */
    public static native boolean isBinningMemorySufficient(GpuParameters ACFInputParams);

    /**
     * Native method. Do binning given a 1D array.
     *
     * @param indata         1D float array. Dimension win_star = w_out x pixbinX +
     *                       cfXDistance, hin_star = h_out x pixbinY + cfYDistance
     * @param outdata        output data, by reference.
     * @param ACFInputParams Values pixbinX, pixbinY are required.
     */
    public static native void calcBinning(float[] indata, float[] outdata, GpuParameters ACFInputParams);

    /**
     * Native method. Get data bleach correction given the 1D pixels values.
     *
     * @param pixels         1D float array of pixels values
     * @param outdata        output data, by reference.
     * @param ACFInputParams Values nopit, ave, cfXDistance, cfYDistance, width
     *                       and height are required
     */
    public static native void calcDataBleachCorrection(float[] pixels, float[] outdata, GpuParameters ACFInputParams);

    /**
     * Retrieves intensity data from the image model.
     * <p>
     * This method attempts to extract the data in one shot using {@code getVoxels()};
     * if that fails, it falls back to a manual voxel copy. Depending on the binning factors,
     * it will perform GPU or CPU binning and subtract background if requested.
     * </p>
     *
     * @param xRange        Range of x indices to extract.
     * @param yRange        Range of y indices to extract.
     * @param returnRawData If {@code false}, background is subtracted from the data.
     * @return A 1D array of intensity data.
     */
    public float[] getIntensityData(Range xRange, Range yRange, boolean returnRawData) {

        boolean needBinning = (this.binningX > 1 || this.binningY > 1);
        int intensityWidth = needBinning ? win_star : w_temp;
        int intensityHeight = needBinning ? hin_star : h_temp;

        // Compute total frames & total output size
        int totalFrames = this.framediff;
        int totalSize = intensityWidth * intensityHeight * totalFrames;
        float[] pixels = new float[totalSize];

        // Try to read from the ImagePlus in one shot, if possible
        ImagePlus imp = imageModel.getImage();
        try {
            // startX, startY are in "binned coordinates."
            // Convert them to unbinned absolute coords:
            int absoluteX = xRange.getStart() * binningX;
            int absoluteY = yRange.getStart() * binningY;
            int absoluteZ = (this.firstframe - 1); // zero-based slice index if needed

            imp.getStack().getVoxels(absoluteX, absoluteY, absoluteZ, intensityWidth, // width
                    intensityHeight, // height
                    totalFrames, // depth
                    pixels // destination
            );
        } catch (Exception e) {
            IJ.log("getVoxels() failed, switching to manual copy. Reason: " + e.getMessage());
            manualVoxelCopy(imp, pixels, xRange, yRange, totalFrames);
        }

        // Check if we need binning (only if binningX or binningY > 1)
        if (needBinning) {
            // Check if we can bin on GPU
            // (This is a placeholder—use your own logic to decide)
            if (canBinOnGpu()) {
                pixels = gpuBinning(pixels);
            } else {
                pixels = cpuBinning(pixels);
            }
        }

        // (D) If returnRawData = false, subtract background
        if (!returnRawData) {
            doBackgroundSubtraction(pixels);
        }

        // (E) Done!
        return pixels;
    }

    /**
     * Fallback method: Manually copies voxel data from the image stack using a triple-nested loop.
     *
     * @param imp         The ImagePlus object containing the image stack.
     * @param dest        Destination array to store copied data.
     * @param xRange      Range for the X dimension.
     * @param yRange      Range for the Y dimension.
     * @param totalFrames Total number of frames to process.
     */
    private void manualVoxelCopy(ImagePlus imp, float[] dest, Range xRange, Range yRange, int totalFrames) {
        int widthTemp = this.w_temp;
        int heightTemp = this.h_temp;
        // For each frame/time index from firstframe..lastframe
        for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
            int z = (this.firstframe - 1) + frameIndex;
            // for each binned y
            for (int by = 0; by < heightTemp; by++) {
                int absY = (yRange.getStart() * binningY) + by;
                // for each binned x
                for (int bx = 0; bx < widthTemp; bx++) {
                    int absX = (xRange.getStart() * binningX) + bx;

                    // Read from the stack voxel by voxel
                    float val = (float) imp.getStack().getVoxel(absX, absY, z);
                    // Store it in the correct location in 'dest'
                    int index = frameIndex * (widthTemp * heightTemp) + by * widthTemp + bx;
                    dest[index] = val;
                }
            }
        }
    }

    /**
     * Performs GPU-based binning using native methods.
     *
     * @param source 1D array of source pixel values.
     * @return Binned pixel data.
     */
    private float[] gpuBinning(float[] source) {
        float[] result = new float[w_temp * h_temp * framediff];
        calcBinning(source, result, this);
        return result;
    }

    /**
     * Performs CPU-based binning on the source pixel data.
     *
     * @param source 1D array of source pixel values.
     * @return Binned pixel data.
     */
    private float[] cpuBinning(float[] source) {
        IJ.log("Performing CPU-based binning...");
        float[] result = new float[w_temp * h_temp * framediff];

        // For each frame
        for (int frame = 0; frame < framediff; frame++) {
            // For each output y in [0..h_temp-1]
            for (int oy = 0; oy < h_temp; oy++) {
                // For each output x in [0..w_temp-1]
                for (int ox = 0; ox < w_temp; ox++) {
                    float sum = 0f;
                    // Sum over the bin
                    for (int by = 0; by < binningY; by++) {
                        for (int bx = 0; bx < binningX; bx++) {
                            // The input index inside the source
                            int inY = oy + by;
                            int inX = ox + bx;
                            int inIndex = frame * (w_temp * h_temp) + inY * w_temp + inX;
                            sum += source[inIndex];
                        }
                    }
                    // The output index in the result array
                    int outIndex = frame * (w_temp * h_temp) + oy * w_temp + ox;
                    result[outIndex] = sum;
                }
            }
        }
        return result;
    }

    /**
     * Checks if GPU-based binning is possible by evaluating the size limit and GPU memory constraints.
     *
     * @return {@code true} if GPU binning is possible; {@code false} otherwise.
     */
    private boolean canBinOnGpu() {
        boolean withinSizeLimit = (w_temp * h_temp * framediff) < 96 * 96 * 50000;
        return withinSizeLimit && isBinningMemorySufficient(this);
    }

    /**
     * Performs background subtraction on the given pixels array.
     * <p>
     * For each pixel (and its sub-bins), subtracts the corresponding background value.
     * </p>
     *
     * @param pixels The array of intensity values (size = w_temp * h_temp * framediff).
     */
    private void doBackgroundSubtraction(float[] pixels) {
        for (int z = 0; z < framediff; z++) {
            for (int x = 0; x < w_temp; x++) {
                for (int y = 0; y < h_temp; y++) {
                    int outIndex = z * (w_temp * h_temp) + y * w_temp + x;

                    // Subtract for each sub-bin
                    for (int bx = 0; bx < binningX; bx++) {
                        for (int by = 0; by < binningY; by++) {
                            int background = imageModel.getBackgroundValue(z + firstframe, x + bx, y + by, 1);
                            pixels[outIndex] -= background;
                        }
                    }
                }
            }
        }
    }

    /**
     * Calculates bleach correction parameters using Gpufit.
     *
     * @param pixels Intensity data to fit
     * @return Array of polynomial coefficients
     */
    public double[] calculateBleachCorrectionParams(float[] pixels) {
        if (!bleachcorr_gpu) {
            return new double[0]; // No correction needed
        }

        int numberFits = w_temp * h_temp;
        float[] datableachCorrection = new float[w_temp * h_temp * nopit];

        // Calculate averaged intensity traces
        calcDataBleachCorrection(pixels, datableachCorrection, this);

        // Gpufit setup
        int numberPoints = nopit;
        Model model = Model.LINEAR_1D;
        Estimator estimator = Estimator.LSE;

        Boolean[] parametersToFit = new Boolean[model.numberParameters];
        parametersToFit[0] = true; // Offset
        for (int i = 1; i < model.numberParameters; i++) {
            parametersToFit[i] = i < bleachcorr_order;
        }

        // Initialize parameters (offset = last point)
        float[] initialParams = new float[numberFits * model.numberParameters];
        for (int i = 0; i < numberFits; i++) {
            int offset = i * model.numberParameters;
            initialParams[offset] = datableachCorrection[(i + 1) * nopit - 1];
            for (int j = 1; j < model.numberParameters; j++) {
                initialParams[offset + j] = 0f;
            }
        }

        // Time points for fitting
        float[] intTime = new float[nopit];
        for (int z = 0; z < nopit; z++) {
            intTime[z] = (float) (frametime * (z + 0.5) * ave);
        }

        float[] weights = new float[numberFits * numberPoints];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 1.0f;
        }

        // Fit using Gpufit
        GpuFitModel fitModel = new GpuFitModel(numberFits, numberPoints, true, model, GpuFitModel.TOLERANCE,
                GpuFitModel.FIT_MAX_ITERATIONS, bleachcorr_order, parametersToFit, estimator, nopit * Float.SIZE / 8);
        fitModel.data.put(datableachCorrection);
        fitModel.weights.put(weights);
        fitModel.initialParameters.put(initialParams);
        fitModel.userInfo.put(intTime);

        FitResult fitResult = Gpufit.fit(fitModel);

        // Extract parameters
        double[] bleachCorrParams = new double[numberFits * bleachcorr_order];
        for (int i = 0; i < numberFits; i++) {
            for (int p = 0; p < bleachcorr_order; p++) {
                bleachCorrParams[i * bleachcorr_order + p] = fitResult.parameters.get(i * model.numberParameters + p);
            }
        }

        return bleachCorrParams;
    }

    /**
     * Applies bleach correction to the intensity data.
     *
     * @param pixels           Intensity data to correct
     * @param bleachCorrParams Polynomial coefficients
     */
    public void applyBleachCorrection(float[] pixels, double[] bleachCorrParams) {
        if (!bleachcorr_gpu || bleachCorrParams.length == 0) {
            return;
        }

        for (int y = 0; y < h_temp; y++) {
            for (int x = 0; x < w_temp; x++) {
                for (int z = 0; z < framediff; z++) {
                    double corfunc = 0;
                    int idx = (y * w_temp + x) * bleachcorr_order;
                    for (int p = 0; p < bleachcorr_order; p++) {
                        corfunc += bleachCorrParams[idx + p] * Math.pow(frametime * (z + 1.5), p);
                    }
                    int pixelIdx = z * w_temp * h_temp + y * w_temp + x;
                    float offset = (float) bleachCorrParams[idx];
                    float val = pixels[pixelIdx];
                    pixels[pixelIdx] =
                            (float) (val / Math.sqrt(corfunc / offset) + offset * (1 - Math.sqrt(corfunc / offset)));
                }
            }
        }
    }
}
