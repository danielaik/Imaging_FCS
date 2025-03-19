package fiji.plugin.imaging_fcs.new_imfcs.enums;

/**
 * Enumerates possible background subtraction methods for image data.
 * Each constant holds a display-friendly string for UI elements.
 */
public enum BackgroundMode implements DisplayNameEnum {

    /**
     * Subtract a single constant background value.
     */
    CONSTANT_BACKGROUND("Constant Background"),

    /**
     * Subtract the frame-specific minimum, computed per frame.
     */
    MIN_FRAME_BY_FRAME("Min frame by frame"),

    /**
     * Subtract one global minimum computed across the entire image stack.
     */
    MIN_PER_IMAGE_STACK("Min per image stack"),

    /**
     * Subtract minimum per pixel, computed across all frames (pixelwise).
     */
    MIN_PIXEL_WISE_PER_IMAGE_STACK("Min Pixel wise per image stack"),

    /**
     * Subtract values from a loaded background image.
     */
    LOAD_BGR_IMAGE("Load BGR image");

    private final String displayName;

    /**
     * Constructs a BackgroundMode with the specified display name.
     *
     * @param displayName the human-readable label for this subtraction mode
     */
    BackgroundMode(String displayName) {
        this.displayName = displayName;
    }

    /**
     * Returns the human-readable label for this background subtraction mode.
     */
    @Override
    public String getDisplayName() {
        return displayName;
    }

    /**
     * Returns the display name as the default string representation.
     */
    @Override
    public String toString() {
        return displayName;
    }
}