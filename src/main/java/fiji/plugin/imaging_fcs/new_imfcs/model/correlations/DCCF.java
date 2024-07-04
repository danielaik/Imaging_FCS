package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import ij.IJ;
import ij.ImagePlus;

import java.awt.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * The DCCF class provides methods to compute the Directional Cross-Correlation Function (DCCF)
 * for an image using a specified direction and experimental settings.
 */
public final class DCCF {
    /**
     * Private constructor to prevent instantiation.
     */
    private DCCF() {
    }

    /**
     * Computes the DCCF for the given image and direction using the specified correlator and settings.
     *
     * @param correlator    The correlator used to compute the pixel correlations.
     * @param img           The image to be processed.
     * @param directionName The name of the direction for DCCF computation.
     * @param settings      The experimental settings model.
     * @return A 2D array representing the computed DCCF values.
     */
    public static double[][] dccf(Correlator correlator, ImagePlus img, String directionName,
                                  ExpSettingsModel settings) {
        // Retrieve the direction object based on the direction name
        Direction direction = Direction.fromName(directionName);

        // Calculate the useful area and pixel binning based on the settings
        Dimension usefulArea = settings.getUsefulArea(img);
        Point pixelBinning = settings.getPixelBinning();

        // Define the lengths of the X and Y dimensions
        int lenX = usefulArea.width - direction.dx + 1;
        int lenY = usefulArea.height - Math.abs(direction.dy) + 1;

        // Determine the start and end Y positions based on the direction
        final int startY = direction.dy != -1 ? 0 : 1;
        final int endY = direction.dy != -1 ? usefulArea.height - direction.dy + 1 : usefulArea.height + 1;

        double[][] dccf = new double[lenX][lenY];
        AtomicInteger progress = new AtomicInteger();

        IntStream.range(0, lenX).parallel().forEach(x -> {
            final PixelModel pixelModel1 = new PixelModel();
            final PixelModel pixelModel2 = new PixelModel();

            for (int y = startY; y < endY; y++) {
                int x1 = x * pixelBinning.x;
                int x2 = (x + direction.dx) * pixelBinning.x;

                int y1 = y * pixelBinning.y;
                int y2 = (y + direction.dy) * pixelBinning.y;

                correlator.correlatePixelModel(pixelModel1, img, x1, y1, x2, y2, settings.getFirstFrame(),
                        settings.getLastFrame());
                correlator.correlatePixelModel(pixelModel2, img, x2, y2, x1, y1, settings.getFirstFrame(),
                        settings.getLastFrame());

                double[] acf1 = pixelModel1.getAcf();
                double[] acf2 = pixelModel2.getAcf();
                for (int i = 1; i < acf1.length; i++) {
                    dccf[x][y - startY] += acf1[i] - acf2[i];
                }
            }

            // update progress
            IJ.showProgress(progress.incrementAndGet(), lenX);
        });

        correlator.setDccf(directionName, dccf);
        return dccf;
    }

    /**
     * The Direction enum represents possible directions for DCCF computation,
     * including X, Y, and diagonal directions.
     */
    private enum Direction {
        X_DIRECTION(Constants.X_DIRECTION, 1, 0),
        Y_DIRECTION(Constants.Y_DIRECTION, 0, 1),
        DIAGONAL_UP(Constants.DIAGONAL_UP_DIRECTION, 1, -1),
        DIAGONAL_DOWN(Constants.DIAGONAL_DOWN_DIRECTION, 1, 1);

        final int dx, dy;
        final String name;

        /**
         * Constructs a Direction enum with the specified name and direction vectors.
         *
         * @param name The name of the direction.
         * @param dx   The change in the X direction.
         * @param dy   The change in the Y direction.
         */
        Direction(String name, int dx, int dy) {
            this.name = name;
            this.dx = dx;
            this.dy = dy;
        }

        /**
         * Retrieves the Direction enum corresponding to the given name.
         *
         * @param name The name of the direction.
         * @return The corresponding Direction enum.
         * @throws IllegalArgumentException if the name does not match any direction.
         */
        static Direction fromName(String name) {
            for (Direction direction : values()) {
                if (direction.name.equals(name)) {
                    return direction;
                }
            }

            throw new IllegalArgumentException("Invalid direction name: " + name);
        }
    }
}
