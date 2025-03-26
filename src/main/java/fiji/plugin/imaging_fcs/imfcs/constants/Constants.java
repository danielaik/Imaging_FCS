package fiji.plugin.imaging_fcs.imfcs.constants;

import java.awt.*;

public final class Constants {
    // define constants
    public static final String PANEL_FONT = "SansSerif";

    public static final int PANEL_FONT_SIZE = 12;
    public static final Point MAIN_PANEL_POS = new Point(10, 125);
    public static final Dimension MAIN_PANEL_DIM = new Dimension(410, 370);
    public static final int TEXT_FIELD_COLUMNS = 8;

    // conversion
    public static final double PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR = Math.pow(10, 6);
    public static final double DIFFUSION_COEFFICIENT_BASE = Math.pow(10, 12);
    public static final double NANO_CONVERSION_FACTOR = Math.pow(10, 9);

    // constants
    public static final double REFRACTIVE_INDEX = 1.3333; // refractive index of water;
    public static final double PI = Math.PI;
    public static final double SQRT_PI = Math.sqrt(PI);

    // Private constructor to prevent instantiation
    private Constants() {
    }
}
