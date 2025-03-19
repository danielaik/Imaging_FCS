package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.enums.BackgroundMode;
import fiji.plugin.imaging_fcs.new_imfcs.enums.FilterMode;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageCanvas;
import ij.gui.ImageWindow;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.process.ImageProcessor;

import java.awt.*;
import java.util.EventListener;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * The ImageModel class encapsulates data and operations for an image.
 */
public final class ImageModel {
    private static final double ZOOM_FACTOR = 250;
    private static final double MAX_ZOOM = 25.0;
    private final BackgroundModel backgroundModel;
    private ImagePlus image;
    private String directory, imagePath, fileName;
    private int width = -1;
    private int height = -1;

    private boolean[][] filterArray = null;

    /**
     * Constructs an ImageModel instance with no image loaded.
     */
    public ImageModel(ExpSettingsModel settings, Runnable resetCallback) {
        this.backgroundModel = new BackgroundModel(settings, resetCallback);
        this.image = null;
    }

    /**
     * Adjusts the scale of the provided ImagePlus object.
     * The image is scaled to fit within a specified zoom factor while maintaining its aspect ratio.
     *
     * @param image The ImagePlus object to be scaled.
     */
    public static void adaptImageScale(ImagePlus image) {
        int maxDim = Math.max(image.getWidth(), image.getHeight());
        double zoomFactor = ZOOM_FACTOR / maxDim;

        // Cap zoom between 100% and MAX_ZOOM (e.g., 3200%)
        zoomFactor = Math.max(1.0, Math.min(zoomFactor, MAX_ZOOM));

        // Apply settings
        IJ.run(image, "Original Scale", "");
        String options =
                String.format("zoom=%.1f x=%d y=%d", zoomFactor * 100, image.getWidth() / 2, image.getHeight() / 2);
        IJ.run(image, "Set... ", options);
        IJ.run(image, "In [+]", "");
    }

    /**
     * Validates that the provided ImagePlus object is of the correct type (GRAY16).
     *
     * @param image The ImagePlus object to check.
     * @throws RuntimeException if the image type is not GRAY16.
     */
    public static void checkImage(ImagePlus image) {
        if (image.getType() != ImagePlus.GRAY16) {
            throw new RuntimeException("Only GRAY16 Tiff stacks supported");
        }
    }

    /**
     * Checks if two ImagePlus objects have different sizes.
     *
     * @param img1 The first ImagePlus object.
     * @param img2 The second ImagePlus object.
     * @return True if the images are not the same size, false otherwise.
     */
    public static boolean areNotSameSize(ImagePlus img1, ImagePlus img2) {
        return img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() ||
                img1.getStackSize() != img2.getStackSize();
    }

    /**
     * Converts the object’s state into a map representation.
     * The map includes key-value pairs for image properties such as the image path, dimensions,
     * and background details.
     *
     * @return a map containing the state of the object, including image properties and background details.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("Image path", imagePath);
        data.put("Image width", getWidth());
        data.put("Image height", getHeight());
        data.put("Background", backgroundModel.getConstantBackground1());
        data.put("Background 2", backgroundModel.getConstantBackground2());
        if (backgroundModel.getBackgroundImage() != null) {
            data.put("Background file", backgroundModel.getBackgroundImage().getOriginalFileInfo().getFilePath());
        } else {
            data.put("Background file", "");
        }

        return data;
    }

    /**
     * Populates the object’s state from a map representation.
     * It reads values from the map to set properties such as background details and attempts
     * to open the background image using the provided file path.
     *
     * @param data a map containing key-value pairs representing the object’s state, including image properties
     *             and background details.
     */
    public void fromMap(Map<String, Object> data) {
        String backgroundPath = data.get("Background file").toString();
        if (!backgroundPath.isEmpty()) {
            ImagePlus backgroundImage = IJ.openImage(backgroundPath);
            if (backgroundImage == null) {
                IJ.log("Failed to open background image at path: " + backgroundPath);
            } else {
                backgroundModel.loadBackgroundImage(image, backgroundImage);
                backgroundModel.setMode(BackgroundMode.LOAD_BGR_IMAGE);
                backgroundModel.computeBackground(image);
            }
        }

        width = Integer.parseInt(data.get("Image width").toString());
        height = Integer.parseInt(data.get("Image height").toString());
        backgroundModel.setConstantBackground1(data.get("Background").toString());
        backgroundModel.setConstantBackground2(data.get("Background 2").toString());
    }

    /**
     * Checks if an image is currently loaded in the model.
     *
     * @return True if an image is loaded, false otherwise.
     */
    public boolean isImageLoaded() {
        return image != null && image.getWindow() != null;
    }

    /**
     * Checks if a background image is currently loaded in the model.
     *
     * @return True if a background image is loaded, false otherwise.
     */
    public boolean isBackgroundLoaded() {
        return backgroundModel.getBackgroundImage() != null;
    }

    /**
     * Removes the extension from a given file name.
     *
     * @param fileName the original file name
     * @return the file name without the extension
     */
    private String removeExtension(String fileName) {
        int lastDotIndex = fileName.lastIndexOf('.');
        if (lastDotIndex > 0) {
            return fileName.substring(0, lastDotIndex);
        }
        return fileName; // Return the original file name if no extension is found
    }

    /**
     * Extracts and stores the path and file name from the ImagePlus object's FileInfo.
     */
    private void retrieveImagePath() {
        this.directory = image.getOriginalFileInfo().directory;
        this.imagePath = image.getOriginalFileInfo().getFilePath();
        this.fileName = removeExtension(image.getOriginalFileInfo().fileName);
    }

    /**
     * Loads an image into the model, checking its type and extracting its path and file name.
     * Only images of type GRAY16 are supported.
     *
     * @param image The ImagePlus object to load into the model.
     * @throws RuntimeException if the image is not of type GRAY16 or if the size doesn't match with the background.
     */
    public void loadImage(ImagePlus image, String simulationName) {
        checkImage(image);

        // if a background image is loaded, check that there are the same format
        if (backgroundModel.getBackgroundImage() != null &&
                areNotSameSize(image, backgroundModel.getBackgroundImage())) {
            throw new RuntimeException("Image is not the same size as the background image");
        }

        // If an image is already loaded, unload it
        if (isImageLoaded()) {
            unloadImage();
        }

        this.image = image;
        this.width = image.getWidth();
        this.height = image.getHeight();

        if (simulationName == null) {
            retrieveImagePath();
        } else {
            this.directory = "";
            this.imagePath = "";
            this.fileName = simulationName;
        }
    }

    /**
     * Unloads the current image from the model.
     */
    public void unloadImage() {
        if (image == null) {
            return;
        }

        if (image.getOverlay() != null) {
            image.deleteRoi();
            image.getOverlay().clear();
            image.setOverlay(null);
        }

        if (isImageLoaded()) {
            ImageCanvas canvas = image.getCanvas();
            removeListeners(canvas.getMouseListeners(), canvas::removeMouseListener);
            removeListeners(canvas.getKeyListeners(), canvas::removeKeyListener);
        }

        image.close();

        image = null;

        // reset filter array
        filterArray = null;
    }

    /**
     * Removes listeners from the provided array using the specified removal function.
     *
     * @param listeners              The array of listeners to remove.
     * @param removeListenerFunction The function to remove each listener.
     * @param <T>                    The type of listener.
     */
    private <T extends EventListener> void removeListeners(T[] listeners, Consumer<T> removeListenerFunction) {
        for (T listener : listeners) {
            removeListenerFunction.accept(listener);
        }
    }

    /**
     * Loads a background image into the model, checking its type and computing statistics.
     *
     * @param backgroundImage The ImagePlus object to load as the background.
     * @return True if the background image was successfully loaded, false otherwise.
     */
    public boolean loadBackgroundImage(ImagePlus backgroundImage) {
        return backgroundModel.loadBackgroundImage(image, backgroundImage);
    }

    /**
     * Validates if the given ROI is within the image bounds.
     *
     * @param roi The Region of Interest to check.
     * @return {@code true} if the ROI is within bounds, {@code false} otherwise.
     */
    public boolean isROIValid(Roi roi) {
        Rectangle rect = roi.getBounds();

        return rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= getWidth() && rect.y + rect.height <= getHeight();
    }

    /**
     * Displays the image in a new window.
     */
    public void show() {
        image.show();
    }


    /**
     * Sets the filter array for marking pixels that do not meet the specified criteria.
     * Depending on the filter mode (e.g., intensity or mean), the pixel values are compared
     * against the provided lower and upper limits over a range of frames.
     *
     * @param mode       The filtering mode (NO_FILTER, FILTER_INTENSITY, FILTER_MEAN).
     * @param lowerLimit The lower intensity limit for the filter.
     * @param upperLimit The upper intensity limit for the filter.
     * @param firstFrame The first frame (1-based) to consider in the image stack.
     * @param lastFrame  The last frame (1-based) to consider in the image stack.
     */
    public void setFilterArray(FilterMode mode, int lowerLimit, int upperLimit, int firstFrame, int lastFrame) {
        if (image == null || mode == FilterMode.NO_FILTER) {
            filterArray = null;
            return;
        }

        filterArray = new boolean[width][height];
        ImageStack stack = image.getStack();

        if (mode == FilterMode.FILTER_INTENSITY) {
            ImageProcessor ip = stack.getProcessor(firstFrame);
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int value = ip.get(x, y);
                    filterArray[x][y] = value < lowerLimit || value > upperLimit;
                }
            }
        } else if (mode == FilterMode.FILTER_MEAN) {
            int numberOfFrames = lastFrame - firstFrame + 1;
            int[] sums = new int[width * height];

            for (int frameIndex = firstFrame; frameIndex <= lastFrame; frameIndex++) {
                short[] pixels = (short[]) stack.getProcessor(frameIndex).getPixels();

                // Add each pixel's value to sums
                for (int idx = 0; idx < pixels.length; idx++) {
                    int val = pixels[idx] & 0xFFFF;  // safe way to get 0..65535
                    sums[idx] += val;
                }
            }

            // 3) Compute mean and set filter flags
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;

                    // integer division or float/double – up to you
                    int meanValue = sums[idx] / numberOfFrames;

                    filterArray[x][y] = (meanValue < lowerLimit || meanValue > upperLimit);
                }
            }
        }
    }

    /**
     * Checks if the pixel at the specified coordinates is filtered.
     *
     * @param x The x-coordinate of the pixel.
     * @param y The y-coordinate of the pixel.
     * @return True if the pixel is marked as filtered, false otherwise.
     */
    public boolean isPixelFiltered(int x, int y) {
        return filterArray != null && filterArray[x][y];
    }

    /**
     * Returns the background value at the specified location and frame, based on the current mode.
     *
     * @param frame   The frame index (1-based for ImageJ).
     * @param x       The x-coordinate of the pixel.
     * @param y       The y-coordinate of the pixel.
     * @param whichBg Identifies which constant background value to return if the mode is constant (1 or 2).
     * @return The background value.
     */
    public int getBackgroundValue(int frame, int x, int y, int whichBg) {
        return backgroundModel.getBackgroundValue(frame, x, y, whichBg);
    }

    // List of getters
    public ImageWindow getWindow() {
        return image.getWindow();
    }

    public ImageCanvas getCanvas() {
        return image.getCanvas();
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public Dimension getDimension() {
        return new Dimension(width, height);
    }

    public int getStackSize() {
        return image.getStackSize();
    }

    public Overlay getOverlay() {
        return image.getOverlay();
    }

    public void setOverlay(Overlay overlay) {
        image.setOverlay(overlay);
    }

    public Roi getRoi() {
        return image.getRoi();
    }

    public void setRoi(Roi roi) {
        image.setRoi(roi);
    }

    public ImagePlus getImage() {
        return image;
    }

    public BackgroundModel getBackgroundModel() {
        return backgroundModel;
    }

    public double getBackgroundCovariance(int x, int y) {
        return backgroundModel.getBackgroundCovariance(x, y);
    }

    public String getDirectory() {
        return directory;
    }

    public String getImagePath() {
        return imagePath;
    }

    public String getFileName() {
        return fileName;
    }
}
