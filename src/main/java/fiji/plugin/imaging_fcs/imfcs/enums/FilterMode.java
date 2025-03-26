package fiji.plugin.imaging_fcs.imfcs.enums;

/**
 * Enumerates the available filtering modes for image processing
 */
public enum FilterMode implements DisplayNameEnum {
    NO_FILTER("No filter"), FILTER_INTENSITY("Intensity"), FILTER_MEAN("Mean");

    private final String displayName;

    /**
     * Constructs a FilterMode with the specified display name.
     *
     * @param displayName the human-readable label for this mode
     */
    FilterMode(String displayName) {
        this.displayName = displayName;
    }

    /**
     * Returns the human-readable label for this filter mode.
     */
    @Override
    public String getDisplayName() {
        return displayName;
    }

    // toString() for combo boxes, logs, etc.
    @Override
    public String toString() {
        return displayName;
    }
}