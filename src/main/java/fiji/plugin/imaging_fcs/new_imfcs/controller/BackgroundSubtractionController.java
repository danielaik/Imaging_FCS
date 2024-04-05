package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.BackgroundSubtractionView;
import ij.IJ;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class BackgroundSubtractionController {
    private final ImageModel imageModel;
    private final BackgroundSubtractionView view;

    public BackgroundSubtractionController(ImageModel imageModel) {
        this.imageModel = imageModel;
        view = new BackgroundSubtractionView(this);
    }

    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    public ActionListener cbBackgroundSubtractionMethodChanged() {
        return (ActionEvent ev) -> {
            String method = ControllerUtils.getComboBoxSelectionFromEvent(ev);
            view.setEnableBackgroundTextField(false);

            switch (method) {
                case "Constant Background":
                    view.setEnableBackgroundTextField(true);
                    break;
                case "Min frame by frame":
                    break;
                case "Min per image stack":
                    break;
                case "Min Pixel wise per image stack":
                    break;
                case "Load BGR image":
                    // Only allows background subtraction before performing bleach correction
                    view.unselectSubtractionAfterBleachCorrection();

                    // Try to load image, revert to other background subtraction method if no background file is loaded
                    boolean loaded = false;
                    try {
                        loaded = imageModel.loadBackgroundImage();
                    } catch (RuntimeException e) {
                        IJ.showMessage(e.getMessage());
                    }

                    if (loaded) {
                        view.updateStatusOnImageLoad(true);
                    } else {
                        ((JComboBox<?>) ev.getSource()).setSelectedIndex(0);
                        view.setEnableBackgroundTextField(true);
                        view.updateStatusOnImageLoad(false);
                    }
                    break;
            }
        };
    }
}
