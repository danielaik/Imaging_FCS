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
            Constants.MAIN_PANEL_POS.y + 100);
    private final Dimension ACF_DIMENSION = new Dimension(200, 200);
    private final Dimension BLOCKING_CURVE_DIMENSION = new Dimension(200, 100);
    private final Point STANDARD_DEVIATION_POSITION = new Point(ACF_POSITION.x + ACF_DIMENSION.width + 115,
            ACF_POSITION.y);
    private final Dimension STANDARD_DEVIATION_DIMENSION = new Dimension(ACF_DIMENSION.width, 50);
    private final Point BLOCKING_CURVE_POSITION =
            new Point(STANDARD_DEVIATION_POSITION.x + STANDARD_DEVIATION_DIMENSION.width + 110,
                    STANDARD_DEVIATION_POSITION.y);
    private final Point COVARIANCE_POSITION = new Point(BLOCKING_CURVE_POSITION.x,
            BLOCKING_CURVE_POSITION.y + BLOCKING_CURVE_DIMENSION.height + 150);
    private final Point INTENSITY_POSITION = new Point(ACF_POSITION.x, ACF_POSITION.y + ACF_DIMENSION.height + 145);
    private final Dimension INTENSITY_DIMENSION = new Dimension(ACF_DIMENSION.width, 50);
    private PlotWindow blockingCurveWindow, acfWindow, standardDeviationWindow, intensityTraceWindow;
    private ImageWindow imgCovarianceWindow;

    public Plots() {
        this.blockingCurveWindow = null;
    }

    private double[] findAdjustedMinMax(double[] array, int len) {
        if (len <= 0) {
            throw new IllegalArgumentException("findAdjustedMinMax: len <= 0");
        }
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 0; i < len; i++) {
            min = Math.min(min, array[i]);
            max = Math.max(max, array[i]);
        }

        // maximum scales need to be 10% larger than maximum value and 10% smaller than minimum value
        min -= min * 0.1;
        max += max * 0.1;

        return new double[]{min, max};
    }

    private PlotWindow plotWindow(Plot plot, PlotWindow window, Point position) {
        // Display the plot in a new window or update the existing one
        if (window == null || window.isClosed()) {
            window = plot.show();
            window.setLocation(position);
        } else {
            window.drawPlot(plot);
        }

        return window;
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

        blockingCurveWindow = plotWindow(plot, blockingCurveWindow, BLOCKING_CURVE_POSITION);
    }

    private Plot getBlockingCurvePlot(double[][] varianceBlocks, int blockCount) {
        double[] minMax = findAdjustedMinMax(varianceBlocks[1], blockCount);
        double minBlock = minMax[0];
        double maxBlock = minMax[1];

        Plot plot = new Plot("blocking", "x", "SD");
        plot.add("line", varianceBlocks[0], varianceBlocks[1]);

        plot.setFrameSize(BLOCKING_CURVE_DIMENSION.width, BLOCKING_CURVE_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(varianceBlocks[0][0] / 2, 2 * varianceBlocks[0][blockCount - 1], minBlock, maxBlock);
        return plot;
    }

    public void plotCovarianceMatrix(int channelNumber, double[][] regularizedCovarianceMatrix) {
        ImagePlus imgCovariance = IJ.createImage("Covariance", "GRAY32", channelNumber - 1, channelNumber - 1, 1);

        ImageProcessor ip = imgCovariance.getProcessor();
        for (int x = 0; x < channelNumber - 1; x++) {
            for (int y = 0; y < channelNumber - 1; y++) {
                ip.putPixelValue(x, y, regularizedCovarianceMatrix[x][y]);
            }
        }

        if (imgCovarianceWindow == null || imgCovarianceWindow.isClosed()) {
            imgCovariance.show();
            imgCovarianceWindow = imgCovariance.getWindow();
            imgCovarianceWindow.setLocation(COVARIANCE_POSITION);
        } else {
            imgCovarianceWindow.setImage(imgCovariance);
        }

        // apply "Spectrum" LUT
        IJ.run(imgCovariance, "Spectrum", "");
        IJ.run(imgCovariance, "Enhance Contrast", "saturated=0.35");

        IJ.run(imgCovariance, "Set... ", "zoom=" + 200 + " x=" + 0 + " y=" + 0);
        // This needs to be used since ImageJ 1.48v to set the window to the right size;
        IJ.run(imgCovariance, "In [+]", "");
    }

    public void plotSingleACF(double[] acf, double[] lagTimes, int channelNumber, int x, int y, Point binning) {
        double[] minMax = findAdjustedMinMax(acf, channelNumber);
        double minScale = minMax[0];
        double maxScale = minMax[1];

        Plot plot = new Plot("CF plot", "tay [s]", "G (tau)");
        plot.setColor(Color.BLUE);
        plot.addPoints(lagTimes, acf, Plot.LINE);
        plot.setFrameSize(ACF_DIMENSION.width, ACF_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(lagTimes[1], 2 * lagTimes[channelNumber - 1], minScale, maxScale);
        plot.setJustification(Plot.CENTER);

        // TODO: create plot label for CCF
        plot.addLabel(0.5, 0, String.format(
                " ACF of (%d, %d) at %dx%d binning.", x, y, binning.x, binning.y));

        plot.draw();

        acfWindow = plotWindow(plot, acfWindow, ACF_POSITION);
    }

    public void plotStandardDeviation(double[] blockStandardDeviation, double[] lagTimes, int channelNumber,
                                      int x, int y) {
        double[] minMax = findAdjustedMinMax(blockStandardDeviation, channelNumber);
        double min = minMax[0];
        double max = minMax[1];

        Plot plot = new Plot("StdDev", "time [s]", "SD");
        plot.setColor(Color.BLUE);
        plot.addPoints(lagTimes, blockStandardDeviation, Plot.LINE);
        plot.setFrameSize(STANDARD_DEVIATION_DIMENSION.width, STANDARD_DEVIATION_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(lagTimes[1], lagTimes[channelNumber - 1], min, max);
        plot.setJustification(Plot.CENTER);
        plot.addLabel(0.5, 0, String.format(" StdDev (%d, %d)", x, y));
        plot.draw();

        // TODO: Add other lines if DC-FCCS(2D) and FCCSDisplay is selected
        standardDeviationWindow = plotWindow(plot, standardDeviationWindow, STANDARD_DEVIATION_POSITION);
    }

    public void plotIntensityTrace(double[] intensityTrace, double[] intensityTime, int numPointsIntensityTrace,
                                   int x, int y) {
        double[] minMax = findAdjustedMinMax(intensityTrace, numPointsIntensityTrace);
        double min = minMax[0];
        double max = minMax[1];

        Plot plot = new Plot("Intensity Trace", "time [s]", "Intensity");
        plot.setFrameSize(INTENSITY_DIMENSION.width, INTENSITY_DIMENSION.height);
        plot.setLimits(intensityTime[1], intensityTime[numPointsIntensityTrace - 1], min, max);
        plot.setColor(Color.BLUE);
        plot.addPoints(intensityTime, intensityTrace, Plot.LINE);
        plot.setJustification(Plot.CENTER);
        plot.addLabel(0.5, 0, String.format(" Intensity Trace (%d, %d)", x, y));
        plot.draw();

        // TODO: add intensity trace 2 if needed

        intensityTraceWindow = plotWindow(plot, intensityTraceWindow, INTENSITY_POSITION);
    }
}
