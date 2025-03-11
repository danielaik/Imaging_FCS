package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.*;
import fiji.plugin.imaging_fcs.new_imfcs.utils.ApplyCustomLUT;
import fiji.plugin.imaging_fcs.new_imfcs.utils.ExcelExporter;
import ij.IJ;
import ij.ImagePlus;
import org.apache.poi.ss.usermodel.Workbook;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Controller class for managing the Number & Brightness (NB) analysis workflow.
 * It connects the NBView and NBModel, handling user interactions and updating the view.
 */
public final class NBController {
    private final NBModel model;

    private final ImageModel imageModel;

    /**
     * Constructs an NBController with the specified models.
     *
     * @param imageModel            the model containing image data
     * @param expSettingsModel      the model containing experiment settings
     * @param options               the model containing options for the analysis
     * @param bleachCorrectionModel the model for bleach correction settings
     */
    public NBController(ImageModel imageModel, ExpSettingsModel expSettingsModel, OptionsModel options,
                        BleachCorrectionModel bleachCorrectionModel) {
        model = new NBModel(imageModel, expSettingsModel, options, bleachCorrectionModel);
        this.imageModel = imageModel;
    }

    /**
     * Creates an ActionListener to handle the NB analysis button press.
     *
     * @return the ActionListener for the NB analysis button
     */
    public ActionListener btnNBPressed() {
        return (ActionEvent ev) -> {
            if (imageModel.isImageLoaded() && imageModel.isBackgroundLoaded()) {
                new BackgroundTaskWorker<Void, Void>(
                        () -> model.performNB(imageModel.getImage(), this::showImage)).execute();
            } else {
                IJ.showMessage("No image and/or background loaded or assigned.");
            }
        };
    }

    /**
     * Displays the given image in the ImageJ window and adapts its scale.
     *
     * @param image    the ImagePlus object to be displayed
     * @param lutColor the custom LUT file to use
     */
    public void showImage(ImagePlus image, String lutColor) {
        image.show();
        ImageModel.adaptImageScale(image);
        ApplyCustomLUT.applyCustomLUT(image, lutColor);
        IJ.run(image, "Enhance Contrast", "saturated=0.35");
    }

    /**
     * Saves the "Number and Brightness" (N&B) data into the provided Excel workbook.
     * This method adds a sheet containing N&B values such as brightness, number, corrected number,
     * and corrected epsilon, using data from the model.
     *
     * @param workbook the Excel workbook to which the N&B sheet will be added
     */
    public void saveExcelSheet(Workbook workbook) {
        ExcelExporter.saveNumberAndBrightnessSheet(workbook, model.getNBB(), model.getNBN());
    }
}