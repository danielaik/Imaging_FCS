package fiji.plugin.imaging_fcs.new_imfcs.model;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageCanvas;
import ij.gui.ImageWindow;

import javax.swing.*;
import java.util.EventListener;
import java.util.function.Consumer;

/**
 * The ImageModel class encapsulates data and operations for an image.
 */
public class ImageModel {
    private static final double ZOOM_FACTOR = 250;
    public ImagePlus image;
    private String imagePath;
    private String fileName;
    private boolean isSimulation;

    /**
     * Constructs an ImageModel instance with no image loaded.
     */
    public ImageModel() {
        this.image = null;
    }

    private boolean isImageLoaded() {
        return image != null && image.getWindow() != null;
    }

    /**
     * Loads an image into the model, checking its type and extracting its path and file name.
     * Only images of type GRAY16 are supported.
     *
     * @param image The ImagePlus object to load into the model.
     * @throws RuntimeException if the image is not of type GRAY16.
     */
    public void loadImage(ImagePlus image, boolean simulation) {
        checkImage(image);
        // If an image is already loaded, unload it
        if (isImageLoaded()) {
            unloadImage();
        }

        this.isSimulation = simulation;
        this.image = image;

        if (!this.isSimulation) {
            getImagePath();
        }
    }

    private void unloadImage() {
        if (image.getOverlay() != null) {
            image.deleteRoi();
            image.getOverlay().clear();
            image.setOverlay(null);
        }

        ImageCanvas canvas = image.getCanvas();
        removeListeners(canvas.getMouseListeners(), canvas::removeMouseListener);
        removeListeners(canvas.getKeyListeners(), canvas::removeKeyListener);

        image = null;
    }

    private <T extends EventListener> void removeListeners(T[] listeners, Consumer<T> removeListenerFunction) {
        for (T listener : listeners) {
            removeListenerFunction.accept(listener);
        }
    }

    public boolean loadBackgroundImage() {
        if (!isImageLoaded()) {
            throw new RuntimeException("No Image loaded, please load an image before loading a background.");
        }

        ImagePlus backgroundImg = null;

        JFileChooser fc = new JFileChooser();
        fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        fc.setMultiSelectionEnabled(false);
        if (fc.showDialog(null, "Choose background file") == JFileChooser.APPROVE_OPTION) {
            backgroundImg = IJ.openImage(fc.getSelectedFile().getAbsolutePath());
            if (backgroundImg == null) {
                IJ.showMessage("Selected file does not exist or it is not an image.");
                return false;
            }

            checkImage(backgroundImg);
        } else {
            IJ.showMessage("No background image loaded.");
            return false;
        }

        if (backgroundImg.getWidth() != image.getWidth() ||
                backgroundImg.getHeight() != image.getWidth() ||
                backgroundImg.getStackSize() != image.getStackSize()) {
            throw new RuntimeException("Background image is not the same size as Image. Background image not loaded");
        }

        // TODO: Compute covariance and row and columns means, just need to see where it is actually used
        return true;
    }

    public void adapt_image_scale() {
        double scimp;
        if (image.getWidth() >= image.getHeight()) {
            scimp = ZOOM_FACTOR / image.getWidth();
        } else {
            scimp = ZOOM_FACTOR / image.getHeight();
        }

        if (scimp < 1.0) {
            scimp = 1.0;
        }
        // Convert scale factor to percentage for the ImageJ command
        scimp *= 100;

        IJ.run(image, "Original Scale", "");
        String options = String.format("zoom=%f x=%d y=%d", scimp, image.getWidth() / 2, image.getHeight() / 2);
        IJ.run(image, "Set... ", options);
        // This workaround addresses a potential bug in ImageJ versions 1.48v and later
        IJ.run("In [+]", "");
    }

    /**
     * Validates that the provided ImagePlus object is of the correct type (GRAY16).
     *
     * @param image The ImagePlus object to check.
     * @throws RuntimeException if the image type is not GRAY16.
     */
    private void checkImage(ImagePlus image) {
        if (image.getType() != ImagePlus.GRAY16) {
            throw new RuntimeException("Only GRAY16 Tiff stacks supported");
        }
    }

    /**
     * Extracts and stores the path and file name from the ImagePlus object's FileInfo.
     */
    private void getImagePath() {
        imagePath = image.getOriginalFileInfo().directory;
        fileName = image.getOriginalFileInfo().fileName;
    }

    /**
     * Displays the image in a new window.
     */
    public void show() {
        image.show();
    }

    // List of getters
    public ImageWindow getWindow() {
        return image.getWindow();
    }

    public ImageCanvas getCanvas() {
        return image.getCanvas();
    }

    public int getHeight() {
        return image.getHeight();
    }

    public int getWidth() {
        return image.getWidth();
    }

    public ImagePlus getImage() {
        return image;
    }
}
