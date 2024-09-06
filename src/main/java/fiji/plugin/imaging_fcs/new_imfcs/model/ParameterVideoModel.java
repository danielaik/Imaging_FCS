package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.utils.ApplyCustomLUT;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Range;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import java.awt.*;

/**
 * Model for handling video parameters and generating parameter videos from image and fit data.
 */
public class ParameterVideoModel {
    private final ExpSettingsModel settings;
    private final ImageModel imageModel;
    private final FitModel interfaceFitModel;
    private int start, end, length, step;
    private boolean saveCFAndFitPVideo = false;
    private String videoName;

    /**
     * Initializes the model with the necessary experiment settings, image data, and fit model.
     *
     * @param settings   the experiment settings model
     * @param imageModel the image model
     * @param fitModel   the fit model
     */
    public ParameterVideoModel(ExpSettingsModel settings, ImageModel imageModel, FitModel fitModel) {
        this.settings = settings;
        this.imageModel = imageModel;
        this.interfaceFitModel = fitModel;
    }

    /**
     * Validates the parameters for video creation.
     */
    private void checkParameters() {
        if (start <= 0 || end <= 0 || end <= start || length <= 0 || step <= 0 || step > (end - start + 1) ||
                length > (end - start + 1)) {
            throw new RuntimeException(
                    "Number error; either negative numbers or length or step size larger than length of frame.");
        }

        videoName = videoName.replaceAll("\\W", "");
        if (videoName.isEmpty()) {
            throw new RuntimeException("Video name is empty or only contains non-word characters.");
        }
    }

    /**
     * Calculates the total number of frames based on the frame range, length, and step size.
     *
     * @return the total number of frames
     */
    private int calculateNumberOfFrames() {
        return (end - start + 1 - length) / step + 1;
    }

    /**
     * Displays the image stack.
     *
     * @param title the title of the image
     * @param stack the image stack to display
     */
    private void displayImageStack(String title, ImageStack stack) {
        ImagePlus img = new ImagePlus(title, stack);
        img.show();
        ImageModel.adaptImageScale(img);

        ApplyCustomLUT.applyCustomLUT(img, "Red Hot");

        IJ.run(img, "Enhance Contrast", "saturated=0.35");
    }

    /**
     * Creates the parameter video based on the current model settings and displays it.
     */
    public void createParameterVideo() {
        // check the parameters and throw exception in case of incorrect parameter
        checkParameters();

        FitModel fitModel = new FitModel(settings, interfaceFitModel);
        fitModel.setFix(true);

        Correlator correlator = new Correlator(settings, fitModel, imageModel);

        Range[] rangesArea = settings.getAllArea(imageModel.getDimension());
        Range xRange = rangesArea[0];
        Range yRange = rangesArea[1];

        ImageStack stackN = new ImageStack(xRange.length() - 1, yRange.length() - 1);
        ImageStack stackD = new ImageStack(xRange.length() - 1, yRange.length() - 1);

        for (int startFrame = start; startFrame < end; startFrame += step) {
            int endFrame = startFrame + length - 1;

            for (int x = xRange.getStart(); x < xRange.getEnd(); x += xRange.getStep()) {
                for (int y = yRange.getStart(); y < yRange.getEnd(); y += yRange.getStep()) {
                    PixelModel currentPixelModel = new PixelModel();
                    correlator.correlatePixelModel(currentPixelModel, imageModel.getImage(), x, y,
                            x + settings.getCCF().width, y + settings.getCCF().height, startFrame, endFrame);
                    fitModel.fit(currentPixelModel, correlator.getLagTimes(),
                            correlator.getRegularizedCovarianceMatrix());


                    Point binningPoint = settings.convertPointToBinning(new Point(x, y));
                    if (currentPixelModel.isFitted() &&
                            !currentPixelModel.toFilter(fitModel, binningPoint.x, binningPoint.y)) {
                        Plots.plotParameterMaps(currentPixelModel, binningPoint, imageModel.getDimension(), null);
                    }
                }
            }

            stackN.addSlice(Plots.imgParam.getStack().getProcessor(1));
            stackD.addSlice(Plots.imgParam.getStack().getProcessor(2));
            Plots.imgParam.close();
        }

        displayImageStack(videoName + " - N", stackN);
        displayImageStack(videoName + " - D", stackD);
    }

    // Getter and setter methods for parameters

    public void setStart(int start) {
        this.start = start;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public void setLength(int length) {
        this.length = length;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public boolean isSaveCFAndFitPVideo() {
        return saveCFAndFitPVideo;
    }

    public void setSaveCFAndFitPVideo(boolean saveCFAndFitPVideo) {
        this.saveCFAndFitPVideo = saveCFAndFitPVideo;
    }

    public void setVideoName(String videoName) {
        this.videoName = videoName;
    }
}
