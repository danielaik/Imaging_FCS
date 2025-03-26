package fiji.plugin.imaging_fcs.imfcs.model.correlations;

import fiji.plugin.imaging_fcs.imfcs.enums.DccfDirection;
import fiji.plugin.imaging_fcs.imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.imfcs.model.PixelModel;
import ij.IJ;
import ij.ImagePlus;

import java.awt.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * The DCCF class provides methods to compute the Delta Cross-Correlation Function (DCCF) for an image using a
 * specified direction and experimental settings. It computes the correlation between p1 and p2,
 * then p2 and p1 and subtract those correlations.
 */
public final class DeltaCCF {
    /**
     * Private constructor to prevent instantiation.
     */
    private DeltaCCF() {
    }

    /**
     * Computes the DCCF for the given image and direction using the specified correlator and settings.
     *
     * @param correlator The correlator used to compute the pixel correlations.
     * @param img        The image to be processed.
     * @param direction  The direction for DCCF computation.
     * @param settings   The experimental settings model.
     * @return A 2D array representing the computed DCCF values.
     */
    public static double[][] dccf(Correlator correlator, ImagePlus img, DccfDirection direction,
                                  ExpSettingsModel settings) {
        // Calculate the useful area and pixel binning based on the settings
        Dimension usefulArea = settings.getUsefulArea(new Dimension(img.getWidth(), img.getHeight()));
        Point pixelBinning = settings.getPixelBinning();

        // Define the lengths of the X and Y dimensions
        int lenX = usefulArea.width - direction.getDx() + 1;
        int lenY = usefulArea.height - Math.abs(direction.getDy()) + 1;

        // Determine the start and end Y positions based on the direction
        final int startY = direction.getDy() != -1 ? 0 : 1;
        final int endY = direction.getDy() != -1 ? usefulArea.height - direction.getDy() + 1 : usefulArea.height + 1;

        double[][] dccf = new double[lenX][lenY];
        AtomicInteger progress = new AtomicInteger();

        IntStream.range(0, lenX).parallel().forEach(x -> {
            final PixelModel pixelModel1 = new PixelModel();
            final PixelModel pixelModel2 = new PixelModel();

            int x1 = x * pixelBinning.x;
            int x2 = (x + direction.getDx()) * pixelBinning.x;

            double[] currentDCCF = dccf[x];

            for (int y = startY; y < endY; y++) {
                int y1 = y * pixelBinning.y;
                int y2 = (y + direction.getDy()) * pixelBinning.y;

                correlator.correlatePixelModel(pixelModel1, img, x1, y1, x2, y2, settings.getFirstFrame(),
                        settings.getLastFrame());
                correlator.correlatePixelModel(pixelModel2, img, x2, y2, x1, y1, settings.getFirstFrame(),
                        settings.getLastFrame());

                double[] acf1 = pixelModel1.getCorrelationFunction();
                double[] acf2 = pixelModel2.getCorrelationFunction();
                for (int i = 1; i < acf1.length; i++) {
                    currentDCCF[y - startY] += acf1[i] - acf2[i];
                }
            }

            // update progress
            IJ.showProgress(progress.incrementAndGet(), lenX);
        });

        correlator.setDccf(direction, dccf);
        return dccf;
    }
}
