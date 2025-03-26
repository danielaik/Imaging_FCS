package fiji.plugin.imaging_fcs.imfcs.model.correlations;

import fiji.plugin.imaging_fcs.imfcs.controller.FitController;
import fiji.plugin.imaging_fcs.imfcs.model.PixelModel;
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
     * Calculates the average correlation function (CF) and its variance for the specified ROI.
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

        double[] averageCF = new double[channelNumber];
        double[] varianceCF = new double[channelNumber];
        // define average array for acf1 and acf2 if FCCSDisp was used
        double[][] averageDCCF = new double[2][channelNumber];
        double[][] varianceDCCF = new double[2][channelNumber];

        int numDCCFPixelsModels = 0;

        for (PixelModel currentPixelModel : pixelModelList) {
            accumulateCorrelationData(currentPixelModel.getCorrelationFunction(), averageCF, varianceCF);

            if (currentPixelModel.getAcf1PixelModel() != null) {
                numDCCFPixelsModels++;
                double[] acf1 = currentPixelModel.getAcf1PixelModel().getCorrelationFunction();
                double[] acf2 = currentPixelModel.getAcf2PixelModel().getCorrelationFunction();

                accumulateCorrelationData(acf1, averageDCCF[0], varianceDCCF[0]);
                accumulateCorrelationData(acf2, averageDCCF[1], varianceDCCF[1]);
            }
        }

        computeAverages(averageCF, varianceCF, numPixelModels);
        PixelModel averagePixelModel = createPixelModel(averageCF, varianceCF);

        // Only compute and set DCCF data if DCCF models are present
        if (numDCCFPixelsModels > 0) {
            computeAverages(averageDCCF[0], varianceDCCF[0], numDCCFPixelsModels);
            computeAverages(averageDCCF[1], varianceDCCF[1], numDCCFPixelsModels);

            averagePixelModel.setAcf1PixelModel(createPixelModel(averageDCCF[0], varianceDCCF[0]));
            averagePixelModel.setAcf2PixelModel(createPixelModel(averageDCCF[1], varianceDCCF[1]));
        }

        return averagePixelModel;
    }

    /**
     * Accumulates the correlation function (CF) values for averaging and variance calculation.
     * This method adds the values from the correlation function (CF) to the corresponding average and variance arrays.
     *
     * @param cf         The correlation function values for the current pixel model.
     * @param averageCF  The array storing the cumulative sum of correlation function values for averaging.
     * @param varianceCF The array storing the cumulative sum of squared correlation function values for variance
     *                   calculation.
     */
    private static void accumulateCorrelationData(double[] cf, double[] averageCF, double[] varianceCF) {
        for (int i = 0; i < cf.length; i++) {
            averageCF[i] += cf[i];
            varianceCF[i] += cf[i] * cf[i];
        }
    }

    /**
     * Computes the average and variance of the correlation function (CF) values.
     * This method calculates the average CF and variance for each channel based on the number of pixel models.
     *
     * @param averageCF      The array storing the cumulative sum of CF values for averaging, updated with the final
     *                       averages.
     * @param varianceCF     The array storing the cumulative sum of squared CF values, updated with the final
     *                       variances.
     * @param numPixelModels The number of pixel models used in the calculation.
     */
    private static void computeAverages(double[] averageCF, double[] varianceCF, int numPixelModels) {
        for (int i = 0; i < averageCF.length; i++) {
            averageCF[i] /= numPixelModels;
            varianceCF[i] = varianceCF[i] / numPixelModels - Math.pow(averageCF[i], 2.0);
        }
    }

    /**
     * Creates a PixelModel object with the given average CF and variance CF.
     *
     * @param averageCF  The array representing the average correlation function.
     * @param varianceCF The array representing the variance of the correlation function.
     * @return A PixelModel object containing the average ACF and its variance.
     */
    private static PixelModel createPixelModel(double[] averageCF, double[] varianceCF) {
        PixelModel averagePixelModel = new PixelModel();
        averagePixelModel.setCorrelationFunction(averageCF);
        averagePixelModel.setVarianceCF(varianceCF);

        return averagePixelModel;
    }
}
