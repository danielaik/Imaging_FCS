package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.NBView;
import ij.IJ;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Objects;

public final class NBController {
    private final NBView view;
    private final ImageModel imageModel;

    public NBController(ImageModel imageModel) {
        view = new NBView(this);
        this.imageModel = imageModel;
    }

    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    public ActionListener cbNBModeChanged() {
        return (ActionEvent ev) -> {
            String mode = ControllerUtils.getComboBoxSelectionFromEvent(ev);
            if (Objects.equals(mode, "G1")) {
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
                // TODO: performNB
            } else {
                IJ.showMessage("No image and/or background loaded or assigned.");
            }
        };
    }
}
