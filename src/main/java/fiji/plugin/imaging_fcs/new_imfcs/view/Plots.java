package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Pair;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Plot;
import ij.gui.PlotWindow;
import ij.process.ImageProcessor;

import java.awt.*;
import java.util.Arrays;

public class Plots {
    private static final Point ACF_POSITION = new Point(
            Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 10, Constants.MAIN_PANEL_POS.y + 100);
    private static final Dimension ACF_DIMENSION = new Dimension(200, 200);
    private static final Dimension BLOCKING_CURVE_DIMENSION = new Dimension(200, 100);
    private static final Point STANDARD_DEVIATION_POSITION =
            new Point(ACF_POSITION.x + ACF_DIMENSION.width + 115, ACF_POSITION.y);
    private static final Dimension STANDARD_DEVIATION_DIMENSION = new Dimension(ACF_DIMENSION.width, 50);
    private static final Point BLOCKING_CURVE_POSITION = new Point(
            STANDARD_DEVIATION_POSITION.x + STANDARD_DEVIATION_DIMENSION.width + 110, STANDARD_DEVIATION_POSITION.y);
    private static final Point COVARIANCE_POSITION =
            new Point(BLOCKING_CURVE_POSITION.x, BLOCKING_CURVE_POSITION.y + BLOCKING_CURVE_DIMENSION.height + 150);
    private static final Point MSD_POSITION =
            new Point(STANDARD_DEVIATION_POSITION.x + STANDARD_DEVIATION_DIMENSION.width + 165, ACF_POSITION.y);
    private static final Point INTENSITY_POSITION =
            new Point(ACF_POSITION.x, ACF_POSITION.y + ACF_DIMENSION.height + 145);
    private static final Dimension INTENSITY_DIMENSION = new Dimension(ACF_DIMENSION.width, 50);
    private static final Dimension MSD_DIMENSION = new Dimension(ACF_DIMENSION);
    private static PlotWindow blockingCurveWindow, acfWindow, standardDeviationWindow, intensityTraceWindow, msdWindow;
    private static ImageWindow imgCovarianceWindow;

    // prevent instantiation
    private Plots() {
    }

    private static Pair<Double, Double> findAdjustedMinMax(double[] array) {
        if (array.length == 0) {
            throw new IllegalArgumentException("findAdjustedMinMax: array is empty");
        }
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 1; i < array.length; i++) {
            min = Math.min(min, array[i]);
            max = Math.max(max, array[i]);
        }

        // maximum scales need to be 10% larger than maximum value and 10% smaller than minimum value
        min -= min * 0.1;
        max += max * 0.1;

        return new Pair<>(min, max);
    }

    private static PlotWindow plotWindow(Plot plot, PlotWindow window, Point position) {
        // Display the plot in a new window or update the existing one
        if (window == null || window.isClosed()) {
            window = plot.show();
            window.setLocation(position);
        } else {
            window.drawPlot(plot);
        }

        return window;
    }

    public static void plotBlockingCurve(double[][] varianceBlocks, int index) {
        Plot plot = getBlockingCurvePlot(varianceBlocks);
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

    private static Plot getBlockingCurvePlot(double[][] varianceBlocks) {
        Pair<Double, Double> minMax = findAdjustedMinMax(varianceBlocks[1]);
        double minBlock = minMax.getLeft();
        double maxBlock = minMax.getRight();

        Plot plot = new Plot("blocking", "x", "SD");
        plot.add("line", varianceBlocks[0], varianceBlocks[1]);

        plot.setFrameSize(BLOCKING_CURVE_DIMENSION.width, BLOCKING_CURVE_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(
                varianceBlocks[0][0] / 2, 2 * varianceBlocks[0][varianceBlocks[0].length - 1], minBlock, maxBlock);
        return plot;
    }

    public static void plotCovarianceMatrix(double[][] regularizedCovarianceMatrix) {
        int len = regularizedCovarianceMatrix.length;
        ImagePlus imgCovariance = IJ.createImage("Covariance", "GRAY32", len, len, 1);

        ImageProcessor ip = imgCovariance.getProcessor();
        for (int x = 0; x < len; x++) {
            for (int y = 0; y < len; y++) {
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

    public static void plotSingleACF(double[] acf, double[] lagTimes, int x, int y, Point binning) {
        Pair<Double, Double> minMax = findAdjustedMinMax(acf);
        double minScale = minMax.getLeft();
        double maxScale = minMax.getRight();

        Plot plot = new Plot("CF plot", "tau [s]", "G (tau)");
        plot.setColor(Color.BLUE);
        plot.addPoints(lagTimes, acf, Plot.LINE);
        plot.setFrameSize(ACF_DIMENSION.width, ACF_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(lagTimes[1], 2 * lagTimes[lagTimes.length - 1], minScale, maxScale);
        plot.setJustification(Plot.CENTER);

        // TODO: create plot label for CCF
        plot.addLabel(0.5, 0, String.format(" ACF of (%d, %d) at %dx%d binning.", x, y, binning.x, binning.y));

        plot.draw();

        acfWindow = plotWindow(plot, acfWindow, ACF_POSITION);
    }

    public static void plotStandardDeviation(double[] blockStandardDeviation, double[] lagTimes, int x, int y) {
        Pair<Double, Double> minMax = findAdjustedMinMax(blockStandardDeviation);
        double min = minMax.getLeft();
        double max = minMax.getRight();

        Plot plot = new Plot("StdDev", "time [s]", "SD");
        plot.setColor(Color.BLUE);
        plot.addPoints(lagTimes, blockStandardDeviation, Plot.LINE);
        plot.setFrameSize(STANDARD_DEVIATION_DIMENSION.width, STANDARD_DEVIATION_DIMENSION.height);
        plot.setLogScaleX();
        plot.setLimits(lagTimes[1], lagTimes[lagTimes.length - 1], min, max);
        plot.setJustification(Plot.CENTER);
        plot.addLabel(0.5, 0, String.format(" StdDev (%d, %d)", x, y));
        plot.draw();

        // TODO: Add other lines if DC-FCCS(2D) and FCCSDisplay is selected
        standardDeviationWindow = plotWindow(plot, standardDeviationWindow, STANDARD_DEVIATION_POSITION);
    }

    public static void plotIntensityTrace(double[] intensityTrace, double[] intensityTime, int x, int y) {
        Pair<Double, Double> minMax = findAdjustedMinMax(intensityTrace);
        double min = minMax.getLeft();
        double max = minMax.getRight();

        Plot plot = new Plot("Intensity Trace", "time [s]", "Intensity");
        plot.setFrameSize(INTENSITY_DIMENSION.width, INTENSITY_DIMENSION.height);
        plot.setLimits(intensityTime[1], intensityTime[intensityTime.length - 1], min, max);
        plot.setColor(Color.BLUE);
        plot.addPoints(intensityTime, intensityTrace, Plot.LINE);
        plot.setJustification(Plot.CENTER);
        plot.addLabel(0.5, 0, String.format(" Intensity Trace (%d, %d)", x, y));
        plot.draw();

        // TODO: add intensity trace 2 if needed

        intensityTraceWindow = plotWindow(plot, intensityTraceWindow, INTENSITY_POSITION);
    }

    public static void plotMSD(double[] msd, double[] lagTimes, int x, int y) {
        Pair<Double, Double> minMax = findAdjustedMinMax(msd);
        double min = minMax.getLeft();
        double max = minMax.getRight();

        double[] msdTime = Arrays.copyOfRange(lagTimes, 0, msd.length);

        Plot plot = new Plot("MSD", "time [s]", "MSD (um^2)");
        plot.setFrameSize(MSD_DIMENSION.width, MSD_DIMENSION.height);
        plot.setLimits(msdTime[1], msdTime[msdTime.length - 1], min, max);
        plot.setColor(Color.BLUE);
        plot.addPoints(msdTime, msd, Plot.LINE);
        plot.setJustification(Plot.CENTER);
        plot.addLabel(0.5, 0, String.format(" MSD (%d, %d)", x, y));
        plot.draw();

        msdWindow = plotWindow(plot, msdWindow, MSD_POSITION);
    }
}
