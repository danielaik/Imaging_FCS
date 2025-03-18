package fiji.plugin.imaging_fcs.new_imfcs.enums;

/**
 * Enumerates possible directions for DCCF computations,
 * each with a display-friendly name for UI elements.
 */
public enum DccfDirection implements DisplayNameEnum {

    /**
     * Autocorrelation in the X direction.
     */
    X_DIRECTION("x direction", 1, 0),

    /**
     * Autocorrelation in the Y direction.
     */
    Y_DIRECTION("y direction", 0, 1),

    /**
     * Diagonal correlation in the "slash" direction.
     */
    DIAGONAL_UP_DIRECTION("diagonal /", 1, -1),

    /**
     * Diagonal correlation in the "backslash" direction.
     */
    DIAGONAL_DOWN_DIRECTION("diagonal \\", 1, 1);

    private final String displayName;
    private final int dx;
    private final int dy;

    /**
     * Constructs a DccfDirection with the specified display name.
     *
     * @param displayName the human-readable label for this direction
     * @param dx          the horizontal offset
     * @param dy          the vertical offset
     */

    DccfDirection(String displayName, int dx, int dy) {
        this.displayName = displayName;
        this.dx = dx;
        this.dy = dy;
    }

    /**
     * Returns the horizontal offset for this direction.
     */
    public int getDx() {
        return dx;
    }

    /**
     * Returns the vertical offset for this direction.
     */
    public int getDy() {
        return dy;
    }

    /**
     * Returns the human-readable label for this DCCF direction.
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