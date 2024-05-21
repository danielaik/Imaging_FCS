package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Plot;
import ij.gui.PlotWindow;
import ij.process.ImageProcessor;

import java.awt.*;

public class Plots {
    private final Point ACF_POSITION = new Point(Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 10,
            Constants.MAIN_PANEL_POS.y + 335);
    private final Dimension ACF_DIMENSION = new Dimension(200, 200);
    private final Dimension BLOCKING_CURVE_DIMENSION = new Dimension(200, 100);
    private final Point STANDARD_DEVIATION_POSITION = new Point(ACF_POSITION.x + ACF_DIMENSION.width + 115,
            ACF_POSITION.y);
    private final Dimension STANDARD_DEVIATION_DIMENSION = new Dimension(ACF_DIMENSION.width, 50);
    private final Point BLOCKING_CURVE_POSITION =
            new Point(STANDARD_DEVIATION_POSITION.x + STANDARD_DEVIATION_DIMENSION.width + 110,
                    STANDARD_DEVIATION_POSITION.y);
    private final Point COVARIANCE_POSITION = new Point(BLOCKING_CURVE_POSITION.x,
            BLOCKING_CURVE_POSITION.y + BLOCKING_CURVE_DIMENSION.width + 50);
    private PlotWindow blockingCurveWindow;
    private ImagePlus imgCovariance;

    public Plots() {
        this.blockingCurveWindow = null;
    }

    public void plotBlockingCurve(double[][] varianceBlocks, int blockCount, int index) {
        Plot plot = getBlockingCurvePlot(varianceBlocks, blockCount);
        plot.setColor(Color.BLUE);
        plot.setJustification(Plot.CENTER);
        plot.addPoints(varianceBlocks[0], varianceBlocks[1], varianceBlocks[2], Plot.CIRCLE);
        plot.draw();

        // Highlight specific points if index is not zero
        if (index != 0) {
            double[][] blockPoints = new double[3][3];
            for (int i = -1; i <= 1; i++) {
                blockPoints[0][i + 1] = varianceBlocks[0][index + i];
                blockPoints[1][i + 1] = varianceBlocks[1][index + i];
                blockPoints[2][i + 1] = varianceBlocks[2][index + i];
            }
            plot.setColor(Color.RED);
            plot.addPoints(blockPoints[0], blockPoints[1], blockPoints[2], Plot.CIRCLE);
            plot.draw();
        }

        // Display the plot in a new window or update the existing one
        if (blockingCurveWindow == null || blockingCurveWindow.isClosed()) {
            blockingCurveWindow = plot.show();
            blockingCurveWindow.setLocation(BLOCKING_CURVE_POSITION);
        } else {
            blockingCurveWindow.drawPlot(plot);
        }
    }

    private Plot getBlockingCurvePlot(double[][] varianceBlocks, int blockCount) {
        double minBlock = Double.MAX_VALUE;
        double maxBlock = Double.MIN_VALUE;

        // Calculate min and max values in varianceBlocks[1]
        for (int i = 0; i < blockCount; i++) {
            minBlock = Math.min(minBlock, varianceBlocks[1][i]);
            maxBlock = Math.max(maxBlock, varianceBlocks[1][i]);
        }

        // adjusting min and max value
        minBlock *= 0.9;
        maxBlock *= 1.1;

        Plot plot = new Plot("blocking", "x", "SD", varianceBlocks[0], varianceBlocks[1]);
        plot.setFrameSize(BLOCKING_CURVE_DIMENSION.width, BLOCKING_CURVE_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(varianceBlocks[0][0] / 2, 2 * varianceBlocks[0][blockCount - 1], minBlock, maxBlock);
        return plot;
    }

    public void plotCovarianceMatrix(int channelNumber, double[][] regularizedCovarianceMatrix) {
        // close covariance window if it exists
        if (imgCovariance != null) {
            imgCovariance.close();
        }

        imgCovariance = IJ.createImage("Covariance", "GRAY32", channelNumber - 1, channelNumber - 1, 1);
        ImageProcessor ip = imgCovariance.getProcessor();
        for (int x = 0; x < channelNumber - 1; x++) {
            for (int y = 0; y < channelNumber - 1; y++) {
                ip.putPixelValue(x, y, regularizedCovarianceMatrix[x][y]);
            }
        }

        imgCovariance.show();
        // apply "Spectrum" LUT
        IJ.run(imgCovariance, "Spectrum", "");
        IJ.run(imgCovariance, "Enhance Contrast", "saturated=0.35");
        ImageWindow imgCovarianceWindow = imgCovariance.getWindow();
        imgCovarianceWindow.setLocation(COVARIANCE_POSITION);

        IJ.run(imgCovariance, "Set... ", "zoom=" + 200 + " x=" + 0 + " y=" + 0);
        // This needs to be used since ImageJ 1.48v to set the window to the right size;
        IJ.run("In [+]", "");
    }
}
