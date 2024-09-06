package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ParameterVideoModel;
import fiji.plugin.imaging_fcs.new_imfcs.utils.ExcelExporter;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.ParameterVideoView;
import ij.IJ;

/**
 * Controls the creation of a parameter video by linking the model and view.
 */
public class ParameterVideoController {
    private final ParameterVideoModel model;
    private final ImageModel imageModel;

    /**
     * Initializes the controller with the settings, image, and fit models.
     *
     * @param settingsModel the experiment settings model
     * @param imageModel    the image model
     * @param fitModel      the fit model
     */
    public ParameterVideoController(ExpSettingsModel settingsModel, ImageModel imageModel, FitModel fitModel) {
        this.imageModel = imageModel;
        model = new ParameterVideoModel(settingsModel, imageModel, fitModel);
    }

    /**
     * Displays the parameter video configuration dialog for the given frame range.
     *
     * @param firstFrame the first frame of the video
     * @param lastFrame  the last frame of the video
     */
    public void showParameterVideoView(int firstFrame, int lastFrame) {
        new ParameterVideoView(firstFrame, lastFrame, model, this::createParameterVideo);
    }

    /**
     * Processes the input from the view and updates the model accordingly.
     *
     * @param view the view containing the user input
     */
    private void createParameterVideo(ParameterVideoView view) {
        model.setStart((int) view.getNextNumber());
        model.setEnd((int) view.getNextNumber());
        model.setLength((int) view.getNextNumber());
        model.setStep((int) view.getNextNumber());
        model.setSaveCFAndFitPVideo(view.getNextBoolean());
        model.setVideoName(view.getNextString());

        if (model.isSaveCFAndFitPVideo()) {
            String excelPath =
                    ExcelExporter.selectExcelFileToSave(String.format("%s_CFAndFit.xlsx", model.getVideoName()),
                            imageModel.getImagePath());
            if (excelPath == null) {
                IJ.showMessage("Error", "No file selected.");
                return;
            }

            model.setExcelPath(excelPath);
        }

        try {
            model.createParameterVideo();
        } catch (RuntimeException e) {
            IJ.showMessage(e.getMessage());
        }
    }
}
