package fiji.plugin.imaging_fcs.imfcs.enums;

/**
 * Enumerates various photobleaching correction methods.
 * Each constant holds a display-friendly string for UI elements.
 */
public enum BleachCorrectionMethod implements DisplayNameEnum {

    /**
     * No bleach correction is applied.
     */
    NO_BLEACH_CORRECTION("none"),

    /**
     * Sliding-window based correction.
     */
    SLIDING_WINDOW("Sliding Window"),

    /**
     * Single-exponential decay correction.
     */
    SINGLE_EXP("Single Exp"),

    /**
     * Double-exponential decay correction.
     */
    DOUBLE_EXP("Double Exp"),

    /**
     * Polynomial fitting based correction.
     */
    POLYNOMIAL("Polynomial"),

    /**
     * Linear-segment piecewise correction.
     */
    LINEAR_SEGMENT("Linear Segment");

    private final String displayName;

    /**
     * Constructs a BleachCorrectionMethod with the specified display name.
     *
     * @param displayName the human-readable label for this method
     */
    BleachCorrectionMethod(String displayName) {
        this.displayName = displayName;
    }

    /**
     * Returns the human-readable label for this bleach correction method.
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