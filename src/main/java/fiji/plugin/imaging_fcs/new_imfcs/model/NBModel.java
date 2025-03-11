package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;

import java.util.Arrays;
import java.util.function.BiConsumer;

/**
 * The NBModel class represents the model for performing number and brightness (N&B) analysis.
 */
public final class NBModel {
    private final ImageModel imageModel;
    private final ExpSettingsModel settings;
    private final OptionsModel options;
    private final BleachCorrectionModel bleachCorrectionModel;

    private double[][] filterArray, NBB, NBN;

    private int frameCount;

    /**
     * Constructs a new NBModel instance with the specified parameters.
     *
     * @param imageModel            The image model containing image data.
     * @param settings              The experimental settings model.
     * @param options               The options model containing analysis options.
     * @param bleachCorrectionModel The bleach correction model.
     */
    public NBModel(ImageModel imageModel, ExpSettingsModel settings, OptionsModel options,
                   BleachCorrectionModel bleachCorrectionModel) {
        this.imageModel = imageModel;
        this.bleachCorrectionModel = bleachCorrectionModel;
        this.settings = settings;
        this.options = options;
    }

    /**
     * Performs number and brightness (N&B) analysis on the specified image.
     *
     * @param img       The ImagePlus object representing the input image.
     * @param showImage A consumer to display the resulting image.
     */
    public void performNB(ImagePlus img, BiConsumer<ImagePlus, String> showImage) {
        int width = img.getWidth();
        int height = img.getHeight();

        filterArray = new double[width][height];
        NBB = new double[width][height];
        NBN = new double[width][height];

        final double filterLL = settings.getFilterLowerLimit();
        final double filterUL = settings.getFilterUpperLimit();

        frameCount = settings.getLastFrame() - settings.getFirstFrame() + 1;

        // If mean is selected we consider all the frames, else just the first frame
        final int framesToConsider = Constants.FILTER_MEAN.equals(settings.getFilter()) ? frameCount : 1;

        if (Constants.NO_FILTER.equals(settings.getFilter())) {
            for (int i = 0; i < width; i++) {
                Arrays.fill(filterArray[i], 1.0);
            }
        } else {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    // This is used when "Mean" is selected, if "Intensity" is selected, we will just the current value
                    double sum = 0;

                    for (int k = settings.getFirstFrame(); k < settings.getFirstFrame() + framesToConsider; k++) {
                        sum += img.getStack().getProcessor(k).getPixelValue(i, j);
                    }

                    // Here we do the mean of the sum. If Intensity is selected, it doesn't change the value (sum / 1)
                    double value = sum / framesToConsider;

                    filterArray[i][j] = (value > filterLL && value < filterUL) ? value : Double.NaN;
                }
            }
        }

        if (options.isUseGpu()) {
            performGpuAnalysis(img);
        } else {
            performCpuAnalysis(img);
        }

        createNBImages(img, showImage);
    }

    private void performGpuAnalysis(ImagePlus img) {
        // TODO: implement me
    }

    /**
     * Performs CPU-based analysis of the fluorescence image.
     *
     * @param img The ImagePlus object representing the input image.
     */
    private void performCpuAnalysis(ImagePlus img) {
        for (int i = 0; i < img.getWidth(); i++) {
            for (int j = 0; j < img.getHeight(); j++) {
                bleachCorrectionModel.calcIntensityTrace(img, i, j, settings.getFirstFrame(), settings.getLastFrame());
                double[] data = bleachCorrectionModel.getIntensity(img, i, j, 1, settings.getFirstFrame(),
                        settings.getLastFrame());

                double mean = 0.0;
                double meanNextElement = 0.0;
                double covariance = 0.0;

                for (int k = 0; k < frameCount - 1; k++) {
                    mean += data[k];
                    meanNextElement += data[k + 1];
                    covariance += data[k] * data[k + 1];
                }

                mean /= frameCount - 1;
                meanNextElement /= frameCount - 1;
                covariance = covariance / (frameCount - 1) - mean * meanNextElement;

                // note that offset has already been corrected for in getIntensity()
                if (Double.isNaN(filterArray[i][j])) {
                    NBB[i][j] = Double.NaN;
                    NBN[i][j] = Double.NaN;
                } else {
                    NBB[i][j] = (covariance - imageModel.getBackgroundCovariance()[i][j]) / mean;
                    NBN[i][j] = mean / NBB[i][j];
                }
            }
        }
    }

    /**
     * Creates and fills an ImagePlus object with the specified title, width, height, and values.
     *
     * @param title  The title of the image.
     * @param width  The width of the image.
     * @param height The height of the image.
     * @param values The values to fill the image with.
     * @return The created ImagePlus object.
     */
    private ImagePlus createAndFillImage(String title, int width, int height, double[][] values) {
        ImagePlus image = IJ.createImage(title, "GRAY32", width, height, 1);
        ImageProcessor ip = image.getStack().getProcessor(1);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                ip.putPixelValue(i, j, values[i][j]);
            }
        }

        return image;
    }

    /**
     * Creates and displays the number and brightness (N&B) images based on the analysis results.
     *
     * @param img       The ImagePlus object representing the input image.
     * @param showImage A consumer to display the resulting image.
     */
    private void createNBImages(ImagePlus img, BiConsumer<ImagePlus, String> showImage) {
        int height = img.getHeight();
        int width = img.getWidth();

        ImagePlus imageN = createAndFillImage("Num N&B", width, height, NBN);
        ImagePlus imageB = createAndFillImage("Brightness N&B", width, height, NBB);

        showImage.accept(imageN, "Cyan Hot");
        showImage.accept(imageB, "Yellow Hot");
    }

    // Getter and setter.
    public double[][] getNBB() {
        return NBB;
    }

    public double[][] getNBN() {
        return NBN;
    }
}
