package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.enums.BackgroundMode;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.BackgroundSubtractionView;
import ij.IJ;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Controller for handling background subtraction logic in the ImageModel.
 * It connects the ImageModel and the BackgroundSubtractionView.
 */
public class BackgroundSubtractionController {
    private final ImageModel imageModel;
    private final Runnable resetCallback;
    private final BackgroundSubtractionView view;

    /**
     * Constructs a BackgroundSubtractionController with the specified ImageModel.
     * Initializes the view associated with this controller.
     *
     * @param imageModel The ImageModel to be controlled.
     */
    public BackgroundSubtractionController(ImageModel imageModel, Runnable resetCallback) {
        this.imageModel = imageModel;
        view = new BackgroundSubtractionView(this, imageModel);

        this.resetCallback = resetCallback;
    }

    /**
     * Sets the visibility of the view.
     *
     * @param b true to make the view visible, false to hide it.
     */
    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    /**
     * Creates an ActionListener that handles changes to the background subtraction method.
     * The listener updates the ImageModel and the view based on the selected method.
     *
     * @return An ActionListener for handling background subtraction method changes.
     */
    public ActionListener cbBackgroundSubtractionMethodChanged(JComboBox<BackgroundMode> comboBox) {
        return new ActionListener() {
            private BackgroundMode previousSelection = (BackgroundMode) comboBox.getSelectedItem();

            @Override
            public void actionPerformed(ActionEvent ev) {
                BackgroundMode mode = (BackgroundMode) comboBox.getSelectedItem();

                try {
                    if (!previousSelection.equals(mode)) {
                        resetCallback.run();
                        // Update previous selection to current if successful
                        previousSelection = mode;
                    }
                } catch (RejectResetException e) {
                    comboBox.setSelectedItem(previousSelection);
                    return;
                }

                view.setEnableBackgroundTextField(false);
                imageModel.resetBackgroundImage();
                setTfBackground(imageModel.getBackground());
                setTfBackground2(imageModel.getBackground2());

                switch (mode) {
                    case CONSTANT_BACKGROUND:
                        view.setEnableBackgroundTextField(true);
                        break;
                    case MIN_FRAME_BY_FRAME:
                        break;
                    case MIN_PER_IMAGE_STACK:
                        break;
                    case MIN_PIXEL_WISE_PER_IMAGE_STACK:
                        break;
                    case LOAD_BGR_IMAGE:
                        // Only allows background subtraction before performing bleach correction
                        view.unselectSubtractionAfterBleachCorrection();

                        // Try to load image, revert to other background subtraction method if no background file is
                        // loaded
                        boolean loaded = false;
                        try {
                            loaded = imageModel.loadBackgroundImage(IJ.openImage());
                            setTfBackground(imageModel.getBackground());
                            setTfBackground2(imageModel.getBackground2());
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
            }
        };
    }

    /**
     * Dispose the view
     */
    public void dispose() {
        this.view.dispose();
    }

    /**
     * Bring the view to front
     */
    public void toFront() {
        this.view.toFront();
    }

    public void setTfBackground(int background) {
        view.setTfBackground(String.valueOf(background));
    }

    public void setTfBackground2(int background2) {
        view.setTfBackground2(String.valueOf(background2));
    }
}
