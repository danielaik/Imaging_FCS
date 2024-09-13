package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import fiji.plugin.imaging_fcs.new_imfcs.controller.FitController;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import ij.gui.Roi;

import java.awt.*;
import java.util.List;
import java.util.function.Function;

/**
 * The AverageCorrelation class provides methods to calculate the average correlation function
 * for a given set of PixelModel objects within a specified Region of Interest (ROI).
 */
public final class AverageCorrelation {
    // private constructor to prevent instantiation.
    private AverageCorrelation() {
    }

    /**
     * Calculates the average autocorrelation function (ACF) and its variance for the specified ROI.
     * If the ROI is null, the calculation is performed on all valid PixelModel objects in the provided array.
     *
     * @param pixelModels           The 2D array of PixelModel objects.
     * @param roi                   The Region of Interest within which to calculate the average correlation function
     *                              . If null, all PixelModel objects are considered.
     * @param convertPointToBinning A function to convert pixel coordinates to their corresponding binning points.
     * @param pixelBinning          The binning factor applied to the pixel coordinates.
     * @param minimumPosition       The minimum position offset to apply to the pixel coordinates.
     * @param fitController         The controller used to filter pixels based on fitting criteria.
     * @return A PixelModel object representing the average correlation function and its variance.
     * @throws RuntimeException if no pixels are correlated within the selected ROI.
     */
    public static PixelModel calculateAverageCorrelationFunction(PixelModel[][] pixelModels, Roi roi,
                                                                 Function<Point, Point> convertPointToBinning,
                                                                 Point pixelBinning, Point minimumPosition,
                                                                 FitController fitController) {
        List<PixelModel> pixelModelList =
                SelectedPixel.getPixelModelsInRoi(roi, pixelBinning, minimumPosition, convertPointToBinning,
                        pixelModels, fitController);

        int numPixelModels = pixelModelList.size();
        if (numPixelModels == 0) {
            throw new RuntimeException("No pixel are correlated in the ROI selected.");
        }

        int channelNumber = pixelModelList.get(0).getCorrelationFunction().length;

        double[] averageAcf = new double[channelNumber];
        double[] varianceAcf = new double[channelNumber];

        for (PixelModel currentPixelModel : pixelModelList) {
            double[] acf = currentPixelModel.getCorrelationFunction();
            for (int i = 0; i < channelNumber; i++) {
                averageAcf[i] += acf[i];
                varianceAcf[i] += acf[i] * acf[i];
            }
        }

        for (int i = 0; i < channelNumber; i++) {
            averageAcf[i] /= numPixelModels;
            varianceAcf[i] = varianceAcf[i] / numPixelModels - Math.pow(averageAcf[i], 2.0);
        }

        return createPixelModel(averageAcf, varianceAcf);
    }

    /**
     * Creates a PixelModel object with the given average ACF and variance ACF.
     *
     * @param averageAcf  The array representing the average autocorrelation function.
     * @param varianceAcf The array representing the variance of the autocorrelation function.
     * @return A PixelModel object containing the average ACF and its variance.
     */
    private static PixelModel createPixelModel(double[] averageAcf, double[] varianceAcf) {
        PixelModel averagePixelModel = new PixelModel();
        averagePixelModel.setCorrelationFunction(averageAcf);
        averagePixelModel.setVarianceCF(varianceAcf);

        return averagePixelModel;
    }
}
