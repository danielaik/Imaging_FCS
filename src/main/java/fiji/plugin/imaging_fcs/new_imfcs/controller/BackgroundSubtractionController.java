package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.view.BackgroundSubtractionView;
import ij.IJ;
import ij.ImagePlus;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class BackgroundSubtractionController {
    private final BackgroundSubtractionView view;

    public BackgroundSubtractionController() {
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
                    if (!loadBGRFile()) {
                        ((JComboBox<?>) ev.getSource()).setSelectedIndex(0);
                        view.setEnableBackgroundTextField(true);
                        view.updateStatusOnImageLoad(false);
                    } else {
                        view.updateStatusOnImageLoad(true);
                    }
                    break;
            }
        };
    }

    private boolean loadBGRFile() {
        JFileChooser fc = new JFileChooser();
        fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        fc.setMultiSelectionEnabled(false);
        if (fc.showDialog(null, "Choose background file") == JFileChooser.APPROVE_OPTION) {
            ImagePlus background_img = IJ.openImage(fc.getSelectedFile().getAbsolutePath());
            if (background_img == null) {
                IJ.showMessage("Selected file does not exist or it is not an image.");
                return false;
            }
        } else {
            IJ.showMessage("No background image loaded.");
            return false;
        }

        return true;
    }
}
