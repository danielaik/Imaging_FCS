package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;

import java.util.Arrays;
import java.util.function.Consumer;

public final class NBModel {
    private final ImageModel imageModel;
    private final ExpSettingsModel settings;
    private final OptionsModel options;
    private final BleachCorrectionModel bleachCorrectionModel;

    private double[][] filterArray, NBB, NBN, NBNum, NBEpsilon;

    private String mode = "G1";
    private int calibRatio = 2;
    private double s_value = 0.0;

    private int frameCount;

    public NBModel(ImageModel imageModel, ExpSettingsModel settings, OptionsModel options,
                   BleachCorrectionModel bleachCorrectionModel) {
        this.imageModel = imageModel;
        this.bleachCorrectionModel = bleachCorrectionModel;
        this.settings = settings;
        this.options = options;
    }

    // perform number and brightness (N&B) analysis
    public void performNB(ImagePlus img, boolean evaluation, Consumer<ImagePlus> showImage) {
        int width = img.getWidth();
        int height = img.getHeight();

        filterArray = new double[width][height];
        NBB = new double[width][height];
        NBN = new double[width][height];

        if (evaluation) {
            NBEpsilon = new double[width][height];
            NBNum = new double[width][height];
        }

        final double filterLL = settings.getFilterLowerLimit();
        final double filterUL = settings.getFilterUpperLimit();

        frameCount = settings.getLastFrame() - settings.getFirstFrame() + 1;

        // If mean is selected we consider all the frames, else just the first frame
        final int framesToConsider = "Mean".equals(settings.getFilter()) ? frameCount : 1;

        if ("none".equals(settings.getFilter())) {
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
            performCpuAnalysis(img, evaluation);
        }

        if (evaluation) {
            createNBImages(img, showImage);
        }
    }

    private void performGpuAnalysis(ImagePlus img) {

    }

    private void performCpuAnalysis(ImagePlus img, boolean evaluation) {
        if ("G1".equals(mode)) {
            calculateMeanAndCovariance(img);
        } else { // Calibrated
            calculateMeanAndVariance(img, evaluation);
        }
    }

    private void calculateMeanAndCovariance(ImagePlus img) {
        for (int i = 0; i < img.getWidth(); i++) {
            for (int j = 0; j < img.getHeight(); j++) {
                bleachCorrectionModel.calcIntensityTrace(img, i, j, settings.getFirstFrame(), settings.getLastFrame());
                double[] data = bleachCorrectionModel.getIntensity(
                        img, i, j, 1, settings.getFirstFrame(), settings.getLastFrame());

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

    private void calculateMeanAndVariance(ImagePlus img, boolean evaluation) {
        for (int i = 0; i < img.getWidth(); i++) {
            for (int j = 0; j < img.getHeight(); j++) {
                bleachCorrectionModel.calcIntensityTrace(img, i, j, 1, img.getStackSize());
                double[] data = bleachCorrectionModel.getIntensity(
                        img, i, j, 1, 1, img.getStackSize());

                double mean = 0;
                double variance = 0;

                for (double element : data) {
                    mean += element;
                    variance += element * element;
                }

                mean /= data.length;
                variance = variance / data.length - mean * mean;

                // note that offset has already been corrected for in getIntensity()
                if (Double.isNaN(filterArray[i][j])) {
                    NBB[i][j] = Double.NaN;
                    NBN[i][j] = Double.NaN;

                    if (evaluation) {
                        NBNum[i][j] = Double.NaN;
                        NBEpsilon[i][j] = Double.NaN;
                    }
                } else {
                    NBB[i][j] = (variance - imageModel.getBackgroundVariance()[i][j]) / mean;
                    NBN[i][j] = (mean * mean) / (variance - imageModel.getBackgroundVariance()[i][j]);

                    if (evaluation) {
                        NBNum[i][j] = mean / (NBB[i][j] - s_value); // n = (mean - offset) / (B - S)
                        NBEpsilon[i][j] = NBB[i][j] / s_value - 1; // epsilon = B / S - 1
                    }
                }
            }
        }
    }

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

    private void createNBImages(ImagePlus img, Consumer<ImagePlus> showImage) {
        int height = img.getHeight();
        int width = img.getWidth();

        ImagePlus imageN = createAndFillImage("Num N&B", width, height, NBN);
        ImagePlus imageB = createAndFillImage("Brightness N&B", width, height, NBB);

        showImage.accept(imageN);
        showImage.accept(imageB);

        if ("Calibrated".equals(mode)) {
            ImagePlus imageNum = createAndFillImage("Num", width, height, NBNum);
            ImagePlus imageEpsilon = createAndFillImage("Epsilon", width, height, NBEpsilon);

            showImage.accept(imageNum);
            showImage.accept(imageEpsilon);
        }
    }

    public String getMode() {
        return mode;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    public int getCalibRatio() {
        return calibRatio;
    }

    public void setCalibRatio(String calibRatio) {
        this.calibRatio = Integer.parseInt(calibRatio);
    }

    public double getS_value() {
        return s_value;
    }

    public void setS_value(String s_value) {
        this.s_value = Double.parseDouble(s_value);
    }
}
