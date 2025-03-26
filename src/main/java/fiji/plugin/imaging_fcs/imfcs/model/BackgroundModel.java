package fiji.plugin.imaging_fcs.imfcs.model;

import fiji.plugin.imaging_fcs.imfcs.enums.BackgroundMode;
import ij.IJ;
import ij.ImagePlus;

/**
 * Manages the background computations and retrieval for intensity-based analyses.
 * Depending on the {@link BackgroundMode}, it either applies a constant background,
 * computes minimum values per frame/pixel, or calculates statistical information
 * (mean, variance, covariance) from a loaded background image.
 */
public class BackgroundModel {
    private final ExpSettingsModel settings;
    private final Runnable resetCallback;
    private BackgroundMode mode = BackgroundMode.CONSTANT_BACKGROUND;

    private ImagePlus backgroundImage = null;
    private int constantBackground1 = 0;
    private int constantBackground2 = 0;
    private int[] frameMin = null;      // one int per frame
    private int globalMin;
    private int[][] pixelMin;
    private int[][] backgroundMean, backgroundVariance;
    private double[][] backgroundCovariance;

    /**
     * Constructs a BackgroundModel with the specified experiment settings and reset callback.
     *
     * @param settings      The experiment settings model.
     * @param resetCallback A runnable invoked whenever certain background properties change.
     */
    public BackgroundModel(ExpSettingsModel settings, Runnable resetCallback) {
        this.settings = settings;
        this.resetCallback = resetCallback;
    }

    /**
     * Computes background values according to the current {@link BackgroundMode}.
     * Depending on the mode, it calculates minimum values frame-by-frame, pixel-wise,
     * or uses a constant value. If a separate background image is loaded
     * ({@link BackgroundMode#LOAD_BGR_IMAGE}), this method delegates to {@link #computeBackgroundStats()}.
     *
     * @param image The image for which the background is computed. Can be null if the mode is set
     *              to {@link BackgroundMode#LOAD_BGR_IMAGE}.
     */
    public void computeBackground(ImagePlus image) {
        constantBackground1 = 0;
        constantBackground2 = 0;

        if (mode == BackgroundMode.LOAD_BGR_IMAGE) {
            computeBackgroundStats();
            return;
        }

        if (image == null || !image.isVisible()) {
            return;
        }

        switch (mode) {
            case CONSTANT_BACKGROUND:
                int minimumValue = computeMinAcrossWholeStack(image, settings.getFirstFrame(), settings.getLastFrame());
                constantBackground1 = minimumValue;
                constantBackground2 = minimumValue;
                break;

            case MIN_FRAME_BY_FRAME:
                computeMinFrameByFrame(image, settings.getFirstFrame(), settings.getLastFrame());
                break;

            case MIN_PER_IMAGE_STACK:
                globalMin = computeMinAcrossWholeStack(image, settings.getFirstFrame(), settings.getLastFrame());
                break;

            case MIN_PIXEL_WISE_PER_IMAGE_STACK:
                computeMinPixelwise(image, settings.getFirstFrame(), settings.getLastFrame());
                break;
        }
    }

    /**
     * Computes the minimum value for each frame within the specified range and stores it
     * in the {@link #frameMin} array. Used by {@link BackgroundMode#MIN_FRAME_BY_FRAME}.
     *
     * @param image      The source image.
     * @param firstFrame The first frame in the range.
     * @param lastFrame  The last frame in the range.
     */
    private void computeMinFrameByFrame(ImagePlus image, int firstFrame, int lastFrame) {
        int nFrames = lastFrame - firstFrame + 1;

        frameMin = new int[nFrames];

        for (int f = firstFrame; f <= lastFrame; f++) {
            short[] pixels = (short[]) image.getStack().getProcessor(f).getPixels();
            int minVal = Integer.MAX_VALUE; // we'll store as int, cast to double after

            for (short pixel : pixels) {
                int value = pixel & 0xFFFF;
                if (value < minVal) {
                    minVal = value;
                }
            }
            frameMin[f - firstFrame] = minVal;
        }
    }

    /**
     * Determines the minimum pixel value across the specified frame range of the image.
     *
     * @param image      The ImagePlus to analyze.
     * @param firstFrame The first frame to include.
     * @param lastFrame  The last frame to include.
     * @return The minimum pixel value within the frame range.
     */
    private int computeMinAcrossWholeStack(ImagePlus image, int firstFrame, int lastFrame) {
        int globalMin = Integer.MAX_VALUE;

        for (int f = firstFrame; f <= lastFrame; f++) {
            short[] pixels = (short[]) image.getStack().getProcessor(f).getPixels();

            for (short pixel : pixels) {
                int value = pixel & 0xFFFF;
                if (value < globalMin) {
                    globalMin = value;
                }
            }
        }

        return globalMin;
    }

    /**
     * Computes the pixel-wise minimum across all frames in the specified range
     * and stores the result in the {@link #pixelMin} array.
     *
     * @param image      The image to process.
     * @param firstFrame The first frame to include.
     * @param lastFrame  The last frame to include.
     */
    private void computeMinPixelwise(ImagePlus image, int firstFrame, int lastFrame) {
        int width = image.getWidth();
        int height = image.getHeight();
        pixelMin = new int[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                pixelMin[x][y] = Integer.MAX_VALUE;
            }
        }

        for (int f = firstFrame; f <= lastFrame; f++) {
            short[] pixels = (short[]) image.getStack().getProcessor(f).getPixels();

            for (int i = 0; i < pixels.length; i++) {
                int value = pixels[i] & 0xFFFF;
                // recover (x,y) from idx
                int x = i % width;
                int y = i / width;

                if (value < pixelMin[x][y]) {
                    pixelMin[x][y] = value;
                }
            }
        }
    }

    /**
     * Computes mean, variance, and covariance statistics from the loaded {@link #backgroundImage}.
     * This method is used when {@link BackgroundMode#LOAD_BGR_IMAGE} is selected.
     */
    private void computeBackgroundStats() {
        int width = backgroundImage.getWidth();
        int height = backgroundImage.getHeight();
        int stackSize = backgroundImage.getStackSize();
        int wh = width * height;  // total pixels per frame

        // Create 1D accumulation arrays
        int[] sumMean = new int[wh];       // accumulates sum of pixel values
        int[] sumVariance = new int[wh];   // accumulates sum of pixel^2
        int[] sumCurrNoLast = new int[wh]; // accumulates sum of pixel values for frames 1..(stackSize-1)
        int[] sumNext = new int[wh];       // accumulates sum of pixel values for frames 2..stackSize
        int[] sumCov = new int[wh];        // accumulates sum of (currFrameVal * nextFrameVal)

        for (int stackIndex = 1; stackIndex <= stackSize; stackIndex++) {
            // Get the current frame's pixel array
            short[] currPixels = (short[]) backgroundImage.getStack().getProcessor(stackIndex).getPixels();

            // Possibly get the next frame's pixel array (for covariance)
            short[] nextPixels = null;
            if (stackIndex < stackSize) {
                nextPixels = (short[]) backgroundImage.getStack().getProcessor(stackIndex + 1).getPixels();
            }

            // Accumulate sums in one pass
            for (int i = 0; i < wh; i++) {
                // Convert signed short to 0..65535
                int currVal = currPixels[i] & 0xFFFF;

                // Accumulate for mean & variance
                sumMean[i] += currVal;
                sumVariance[i] += currVal * currVal;

                // Covariance requires having a "next" frame
                if (nextPixels != null) {
                    int nextVal = nextPixels[i] & 0xFFFF;

                    // sum of frames except last: add currentVal
                    sumCurrNoLast[i] += currVal;
                    // sum of frames except first: add nextVal
                    sumNext[i] += nextVal;
                    sumCov[i] += currVal * nextVal;
                }
            }
        }

        backgroundMean = new int[width][height];
        backgroundVariance = new int[width][height];
        backgroundCovariance = new double[width][height];

        // Map our 1D accumulations to 2D arrays, and finalize the stats
        for (int i = 0; i < wh; i++) {
            int x = i % width;
            int y = i / width;

            double mean = (double) sumMean[i] / stackSize;
            backgroundMean[x][y] = (int) Math.round(mean);
            backgroundVariance[x][y] = (int) Math.round(((double) sumVariance[i] / stackSize - mean * mean));

            // Covariance only defined if stackSize > 1
            if (stackSize > 1) {
                int stackMinus1 = stackSize - 1;
                double sumC = sumCurrNoLast[i];
                double sumN = sumNext[i];
                double cov = (double) sumCov[i] / stackMinus1 - (sumC / stackMinus1) * (sumN / stackMinus1);
                backgroundCovariance[x][y] = cov;
            }
        }
    }

    /**
     * Returns the background value at the specified location and frame, based on the current mode.
     *
     * @param frame   The frame index (1-based for ImageJ).
     * @param x       The x-coordinate of the pixel.
     * @param y       The y-coordinate of the pixel.
     * @param whichBg Identifies which constant background value to return if the mode is constant (1 or 2).
     * @return The background value.
     */
    public int getBackgroundValue(int frame, int x, int y, int whichBg) {
        switch (mode) {
            case CONSTANT_BACKGROUND:
                if (whichBg == 1) {
                    return constantBackground1;
                } else {
                    return constantBackground2;
                }

            case MIN_FRAME_BY_FRAME:
                return frameMin[frame - settings.getFirstFrame()];

            case MIN_PER_IMAGE_STACK:
                return globalMin;

            case MIN_PIXEL_WISE_PER_IMAGE_STACK:
                return pixelMin[x][y];

            case LOAD_BGR_IMAGE:
                // If you only have one background image, same for both ROI1 & ROI2
                return backgroundMean[x][y];

            default:
                // fallback
                return 0;
        }
    }

    /**
     * Loads a background image for {@link BackgroundMode#LOAD_BGR_IMAGE}, checking compatibility
     * and size. Resets constant backgrounds upon successful loading.
     *
     * @param image           An image to check size compatibility against (optional).
     * @param backgroundImage The candidate background image to load.
     * @return True if successfully loaded; false otherwise.
     */
    public boolean loadBackgroundImage(ImagePlus image, ImagePlus backgroundImage) {
        if (backgroundImage == null) {
            IJ.showMessage("Selected file does not exist or it is not an image.");
            return false;
        }

        ImageModel.checkImage(backgroundImage);

        if (image != null && ImageModel.areNotSameSize(image, backgroundImage)) {
            throw new RuntimeException("Background image is not the same size as Image. Background image not loaded");
        }

        this.backgroundImage = backgroundImage;

        // set background values to 0 if a background is loaded
        constantBackground1 = 0;
        constantBackground2 = 0;
        return true;
    }

    /**
     * Clears the currently loaded background image and resets associated statistics.
     */
    public void resetBackgroundImage() {
        backgroundImage = null;
        backgroundMean = null;
        backgroundVariance = null;
        backgroundCovariance = null;
    }

    public ImagePlus getBackgroundImage() {
        return backgroundImage;
    }

    public double getBackgroundCovariance(int x, int y) {
        return backgroundCovariance[x][y];
    }

    public BackgroundMode getMode() {
        return mode;
    }

    public void setMode(BackgroundMode mode) {
        this.mode = mode;
    }

    public int getConstantBackground1() {
        return constantBackground1;
    }

    public void setConstantBackground1(String constantBackground1) {
        int tmp = Integer.parseInt(constantBackground1);
        resetCallback.run();
        this.constantBackground1 = tmp;
    }

    public int getConstantBackground2() {
        return constantBackground2;
    }

    public void setConstantBackground2(String constantBackground2) {
        int tmp = Integer.parseInt(constantBackground2);
        resetCallback.run();
        this.constantBackground2 = tmp;
    }
}
