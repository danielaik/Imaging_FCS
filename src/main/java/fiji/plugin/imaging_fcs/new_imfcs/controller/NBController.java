package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.NBModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.OptionsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.NBView;
import ij.IJ;
import ij.ImagePlus;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public final class NBController {
    private final NBView view;
    private final NBModel model;

    private final ImageModel imageModel;

    public NBController(ImageModel imageModel, ExpSettingsModel expSettingsModel, OptionsModel options,
                        BleachCorrectionModel bleachCorrectionModel) {
        model = new NBModel(imageModel, expSettingsModel, options, bleachCorrectionModel);
        view = new NBView(this, model);
        this.imageModel = imageModel;
    }

    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    public ActionListener cbNBModeChanged() {
        return (ActionEvent ev) -> {
            String mode = ControllerUtils.getComboBoxSelectionFromEvent(ev);
            model.setMode(mode);

            if ("G1".equals(mode)) {
                view.setEnabledTextFields(false);
            } else {
                view.setEnabledTextFields(true);
                if (!imageModel.isBackgroundLoaded()) {
                    IJ.showMessage("Background not loaded, please load a background first.");
                    view.setEnabledTextFields(false);
                    ((JComboBox<?>) ev.getSource()).setSelectedIndex(0);
                } else {
                    // TODO: camCalibrate
                }
            }
        };
    }

    public ActionListener btnNBPressed() {
        return (ActionEvent ev) -> {
            if (imageModel.isImageLoaded() && imageModel.isBackgroundLoaded()) {
                model.performNB(imageModel.getImage(), true, this::showImage);
            } else {
                IJ.showMessage("No image and/or background loaded or assigned.");
            }
        };
    }

    public void showImage(ImagePlus image) {
        image.show();
        ImageModel.adaptImageScale(image);
    }
}
