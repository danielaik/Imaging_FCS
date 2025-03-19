package fiji.plugin.imaging_fcs.new_imfcs.enums;

/**
 * Enumerates various fit functions methods.
 * Each constant holds a display-friendly string for UI elements.
 */
public enum FitFunctions implements DisplayNameEnum {
    /**
     * DC-FCCS in 2D.
     */
    DC_FCCS_2D("DC-FCCS (2D)"),

    /**
     * ITIR-FCS in 2D.
     */
    ITIR_FCS_2D("ITIR-FCS (2D)"),

    /**
     * A variant of ITIR-FCS (2D), used to indicate a second
     * PSF parameter in FCSFit for FCCS setups.
     */
    ITIR_FCS_2D_2("ITIR-FCS (2D) 2"),

    /**
     * SPIM-FCS in 3D.
     */
    SPIM_FCS_3D("SPIM-FCS (3D)");

    private final String displayName;

    /**
     * Constructs a FitModel with a given display name.
     *
     * @param displayName the human-readable label for this model
     */
    FitFunctions(String displayName) {
        this.displayName = displayName;
    }

    /**
     * Returns the human-readable label for this fit function.
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
