package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import ij.ImagePlus;

import java.util.Arrays;

public class Correlator {
    // minimum number of frames required for the sliding windows; this is used to
    // calculate a useful correlatorq
    private final int SLIDING_WINDOW_MIN_FRAME = 20;
    private final int BLOCK_LAG = 1;
    private final ExpSettingsModel settings;
    private final BleachCorrectionModel bleachCorrectionModel;
    private final Plots plots;
    private int channelNumber;
    private int lagGroupNumber;
    private int correlatorQ;
    private int[] numSamples;
    private int[] lags;
    private int[] sampleTimes;
    private double[] lagTimes;
    private double[] blockVariance;
    private double[] blockStandardDeviation;
    private double[] meanCovariance;
    private double[][] regularizedCovarianceMatrix;


    public Correlator(ExpSettingsModel settings, BleachCorrectionModel bleachCorrectionModel, Plots plots) {
        this.settings = settings;
        this.bleachCorrectionModel = bleachCorrectionModel;
        this.plots = plots;
    }

    private double[][] getIntensityBlock(ImagePlus img, int x, int y, int x2, int y2,
                                         int initialFrame, int finalFrame, int mode) {
        double[] intensityData = bleachCorrectionModel.getIntensity(img, x, y, mode,
                initialFrame, finalFrame);

        double[][] intensityBlock = new double[2][intensityData.length];
        intensityBlock[0] = intensityData;

        if (x != x2 || y != y2) {
            // get intensity for second pixel
            intensityBlock[1] = bleachCorrectionModel.getIntensity(img, x2, y2, 2, initialFrame, finalFrame);
        } else {
            // otherwise perform an auto correlation
            intensityBlock[1] = intensityData;
        }

        return intensityBlock;
    }

    public void correlate(ImagePlus img, int x, int y, int initialFrame, int finalFrame) {
        correlate(img, x, y, x, y, initialFrame, finalFrame);
    }

    public void correlate(ImagePlus img, int x, int y, int x2, int y2, int initialFrame, int finalFrame) {
        int numFrames = finalFrame - initialFrame + 1;
        correlatorQ = settings.getCorrelatorQ();

        if (settings.getBleachCorrection().equals(Constants.BLEACH_CORRECTION_SLIDING_WINDOW)) {
            int numSlidingWindow = numFrames / bleachCorrectionModel.getSlidingWindowLength();
            lagGroupNumber = (int) Math.floor((Math.log(
                    (double) bleachCorrectionModel.getSlidingWindowLength() /
                            (SLIDING_WINDOW_MIN_FRAME + settings.getCorrelatorP()))
                    + 1) / Math.log(2));
            channelNumber = settings.getCorrelatorP() + (lagGroupNumber - 1) * settings.getCorrelatorP() / 2 + 1;

            // allow smaller correlator Q value as minimum but not larger
            correlatorQ = Math.min(correlatorQ, lagGroupNumber);

            for (int i = 0; i < numSlidingWindow; i++) {
                int slidingWindowInitialFrame = i * bleachCorrectionModel.getSlidingWindowLength() + initialFrame;
                int slidingWindowFinalFrame = (i + 1) * bleachCorrectionModel.getSlidingWindowLength() +
                        initialFrame - 1;

                double[][] intensityBlock = getIntensityBlock(img, x, y, x2, y2,
                        slidingWindowInitialFrame, slidingWindowFinalFrame, 1);

                int index = blockTransform(intensityBlock, bleachCorrectionModel.getSlidingWindowLength());
                // calculateCF(intensityBlock, bleachCorrectionModel.getSlidingWindowLength(), index);
            }
        } else {
            // if sliding window is not selected, correlate the full intensity trace
            lagGroupNumber = correlatorQ;
            channelNumber = settings.getCorrelatorP() +
                    (correlatorQ - 1) * settings.getCorrelatorP() / 2 + 1;

            // TODO: check kcf (select 1, 2 or 3)
            int mode = settings.getFitModel().equals(Constants.DC_FCCS_2D) ? 2 : 1;
            double[][] intensityBlock = getIntensityBlock(img, x, y, x2, y2, initialFrame, finalFrame, mode);

            int index = blockTransform(intensityBlock, numFrames);
            // calculateCF(intensityBlock, numFrames, index);
        }
    }

    private int blockTransform(double[][] intensityCorrelation, int numFrames) {
        int blockCount = calculateBlockCount(numFrames);

        double[][] varianceBlocks = new double[3][blockCount];
        double[] lowerQuartile = new double[blockCount];
        double[] upperQuartile = new double[blockCount];

        processBlocks(intensityCorrelation, blockCount, numFrames, varianceBlocks, lowerQuartile, upperQuartile);
        int index = determineLastIndexMeetingCriteria(blockCount, varianceBlocks, lowerQuartile, upperQuartile);

        plots.plotBlockingCurve(varianceBlocks, blockCount, index);

        return index;
    }

    private int calculateBlockCount(int numFrames) {
        calculateParameters(numFrames);
        return (int) Math.floor(Math.log(numSamples[BLOCK_LAG]) / Math.log(2)) - 2;
    }

    private void calculateParameters(int numFrames) {
        lags = new int[channelNumber];
        lagTimes = new double[channelNumber];
        sampleTimes = new int[channelNumber];
        numSamples = new int[channelNumber];

        calculateLags(settings.getCorrelatorP(), settings.getCorrelatorP() / 2);
        calculateSampleTimes(settings.getCorrelatorP(), settings.getCorrelatorP() / 2);
        calculateNumberOfSamples(numFrames);
    }

    private void calculateLags(int numChannelsFirstGroup, int numChannelsHigherGroups) {
        for (int i = 0; i <= numChannelsHigherGroups; i++) {
            lags[i] = i;
            lagTimes[i] = i * settings.getFrameTime();
        }

        for (int group = 1; group <= lagGroupNumber; group++) {
            for (int channel = 1; channel <= numChannelsHigherGroups; channel++) {
                int index = group * numChannelsHigherGroups + channel;
                lags[index] = (int) (Math.pow(2, group - 1) * channel +
                        (numChannelsFirstGroup / 4) * Math.pow(2, group));
                lagTimes[index] = lags[index] * settings.getFrameTime();
            }
        }
    }

    // calculate sampletimes (bin width) for the 0 lagtime kcf
    private void calculateSampleTimes(int numChannelsFirstGroup, int numChannelsHigherGroups) {
        Arrays.fill(sampleTimes, 0, numChannelsFirstGroup + 1, 1);

        for (int group = 2; group <= lagGroupNumber; group++) {
            for (int channel = 1; channel <= numChannelsHigherGroups; channel++) {
                sampleTimes[group * numChannelsHigherGroups + channel] = (int) Math.pow(2, group - 1);
            }
        }
    }

    private void calculateNumberOfSamples(int numFrames) {
        for (int i = 0; i < channelNumber; i++) {
            numSamples[i] = (numFrames - lags[i]) / sampleTimes[i];
        }
    }

    private void binData(int numBinnedDataPoints, double[][] intensityBlock) {
        for (int i = 0; i < numBinnedDataPoints; i++) {
            intensityBlock[0][i] = intensityBlock[0][2 * i] + intensityBlock[0][2 * i + 1];
            intensityBlock[1][i] = intensityBlock[1][2 * i] + intensityBlock[1][2 * i + 1];
        }
    }

    private void processBlocks(double[][] intensityBlock, int blockCount, int numFrames, double[][] varianceBlocks,
                               double[] lowerQuartile, double[] upperQuartile) {
        int currentIncrement = BLOCK_LAG;
        int numBinnedDataPoints = numFrames;
        double[] numProducts = new double[blockCount];

        for (int i = 1; i < channelNumber; i++) {
            // check whether the kcf width has changed
            if (currentIncrement != sampleTimes[i]) {
                // set the current increment accordingly
                currentIncrement = sampleTimes[i];
                // Correct the number of actual data points accordingly
                numBinnedDataPoints /= 2;
                binData(numBinnedDataPoints, intensityBlock);
            }

            if (i == BLOCK_LAG) {
                processCorrelationData(i, blockCount, numFrames, numBinnedDataPoints, intensityBlock, varianceBlocks,
                        numProducts, currentIncrement);
            }
        }

        for (int i = 0; i < blockCount; i++) {
            varianceBlocks[1][i] = Math.sqrt(varianceBlocks[1][i]);
            varianceBlocks[2][i] = varianceBlocks[1][i] / Math.sqrt(2 * (numProducts[i] - 1));
            upperQuartile[i] = varianceBlocks[1][i] + varianceBlocks[2][i];
            lowerQuartile[i] = varianceBlocks[1][i] - varianceBlocks[2][i];
        }
    }

    private double[] calculateMonitors(double numProducts, double[][] intensityBlock, int delay) {
        double directMonitor = 0.0;
        double delayedMonitor = 0.0;

        for (int i = 0; i < numProducts; i++) {
            directMonitor += intensityBlock[0][i];
            delayedMonitor += intensityBlock[1][i + delay];
        }
        directMonitor /= numProducts;
        delayedMonitor /= numProducts;

        return new double[]{directMonitor, delayedMonitor};
    }

    private double[] calculateCorrelations(double numProducts, double[][] intensityBlock, double directMonitor,
                                           double delayedMonitor, double[] products, int delay) {
        double sumProd = 0.0;
        double sumProdSquared = 0.0;

        for (int i = 0; i < numProducts; i++) {
            products[i] = intensityBlock[0][i] * intensityBlock[1][i + delay] -
                    delayedMonitor * intensityBlock[0][i] - directMonitor * intensityBlock[1][i + delay] +
                    delayedMonitor * directMonitor;
            sumProd += products[i];
            sumProdSquared += Math.pow(products[i], 2);
        }

        return new double[]{sumProd, sumProdSquared};
    }

    private void performBlockingOperations(int blockCount, int currentIncrement, double[][] varianceBlocks,
                                           double directMonitor, double delayedMonitor, double[] products,
                                           double[] numProducts) {
        double sumProd, sumProdSquared;

        for (int i = 1; i < blockCount; i++) {
            numProducts[i] = (int) (numProducts[i - 1] / 2);
            sumProd = sumProdSquared = 0.0;
            for (int j = 0; j < numProducts[i]; j++) {
                products[j] = (products[2 * j] + products[2 * j + 1]) / 2;
                sumProd += products[j];
                sumProdSquared += products[j] * products[j];
            }

            // the time of the block curve
            varianceBlocks[0][i] = (currentIncrement * Math.pow(2, i)) * settings.getFrameTime();

            // value of the block curve
            varianceBlocks[1][i] = (sumProdSquared / numProducts[i] - Math.pow(sumProd / numProducts[i], 2)) /
                    (numProducts[i] * Math.pow(directMonitor * delayedMonitor, 2));
        }
    }

    private void processCorrelationData(int i, int blockCount, int numFrames, int numBinnedDataPoints,
                                        double[][] intensityBlock, double[][] varianceBlocks, double[] numProducts,
                                        int currentIncrement) {
        int delay = lags[i] / currentIncrement;
        numProducts[0] = numBinnedDataPoints - delay;

        double[] monitors = calculateMonitors(numProducts[0], intensityBlock, delay);
        double directMonitor = monitors[0];
        double delayedMonitor = monitors[1];

        double[] products = new double[numFrames];
        double[] sumProds = calculateCorrelations(numProducts[0], intensityBlock, directMonitor, delayedMonitor,
                products, delay);
        double sumProd = sumProds[0];
        double sumProdSquared = sumProds[1];

        varianceBlocks[0][0] = currentIncrement * settings.getFrameTime();
        varianceBlocks[1][0] = (sumProdSquared / (numBinnedDataPoints - delay) -
                Math.pow(sumProd / numProducts[0], 2)) / (numProducts[0] * Math.pow(directMonitor * delayedMonitor, 2));

        performBlockingOperations(blockCount, currentIncrement, varianceBlocks, directMonitor, delayedMonitor, products,
                numProducts);
    }

    private int determineLastIndexMeetingCriteria(int blockCount, double[][] varianceBlocks,
                                                  double[] lowerQuartile, double[] upperQuartile) {
        int lastIndexMeetingCriteria = -1;
        int firstOverlapingIndexAfterIncrease = -1;

        for (int i = 0; i < blockCount - 2; i++) {
            // Check if neighboring points have non overlapping error bars
            if (haveNonOverlappingErrorBars(i, upperQuartile, lowerQuartile) &&
                    haveNonOverlappingErrorBars(i + 1, upperQuartile, lowerQuartile)) {
                // Check if these three points are the last triple with increasing differences
                if (isIncreasing(i, varianceBlocks) && isIncreasing(i + 1, varianceBlocks)) {
                    lastIndexMeetingCriteria = i;
                }
            } else if (i < blockCount - 4 && lastIndexMeetingCriteria > firstOverlapingIndexAfterIncrease) {
                // the last two points can't be part of the blocking triple
                firstOverlapingIndexAfterIncrease = i;
            }
        }

        int index = firstOverlapingIndexAfterIncrease + 1;

        if (lastIndexMeetingCriteria == -1 || index < lastIndexMeetingCriteria) {
            // optimal blocking is not possible, use maximal blocking
            if (blockCount > 3) {
                index = blockCount - 3;
            } else {
                index = blockCount - 1;
            }
        }

        return Math.max(index, correlatorQ - 1);
    }

    private boolean haveNonOverlappingErrorBars(int index, double[] upperQuartile, double[] lowerQuartile) {
        return !(upperQuartile[index] > lowerQuartile[index + 1] && upperQuartile[index + 1] > lowerQuartile[index]);
    }

    private boolean isIncreasing(int index, double[][] varianceBlocks) {
        return varianceBlocks[1][index + 1] - varianceBlocks[1][index] > 0;
    }

    private double calculateBlockVariance(double[] products, double directMonitor, double delayedMonitor,
                                          double numProducts) {
        double sumProd = 0.0;
        double sumProdSquared = 0.0;

        for (int i = 0; i < numProducts; i++) {
            // calculate the sum of prod, i.e. the raw correlation value
            sumProd += products[i];
            sumProdSquared += Math.pow(products[i], 2);
        }

        // variance after blocking; extra division by numProduct to obtain SEM
        return (sumProdSquared / numProducts - Math.pow(sumProd / numProducts, 2)) /
                ((numProducts - 1) * Math.pow(directMonitor * delayedMonitor, 2));
    }

    private void calculateMeanCovariance(double[][] products, double[] directMonitors, double[] delayedMonitors,
                                         int minProducts) {
        for (int i = 1; i < channelNumber; i++) {
            for (int j = 0; j < minProducts; j++) {
                meanCovariance[i] = products[i][j] / (directMonitors[i] * delayedMonitors[i]);
            }
            // normalize by the number of products
            meanCovariance[i] /= minProducts;
        }
    }

    private void calculateCovarianceMatrix(double[][] covarianceMatrix, double[][] products, double[] directMonitors,
                                           double[] delayedMonitors, int minProducts) {
        for (int i = 1; i < channelNumber; i++) {
            for (int j = 1; j <= i; j++) {
                for (int k = 0; k < minProducts; k++) {
                    covarianceMatrix[i][j] += (products[i][k] / (directMonitors[i] * delayedMonitors[i]) *
                            (products[j][k] / (directMonitors[j] * delayedMonitors[j]) - meanCovariance[j]));
                }
                covarianceMatrix[i][j] /= (minProducts - 1);
                // lower triangular part is equal to upper triangular part
                covarianceMatrix[j][i] = covarianceMatrix[i][j];
            }
        }
    }

    private double calculateVarianceShrinkageWeight(double[][] covarianceMatrix, double[][] products,
                                                    double[] directMonitors, double[] delayedMonitors,
                                                    int minProducts) {
        double[] diagonalCovarianceMatrix = new double[channelNumber];

        for (int i = 1; i < channelNumber; i++) {
            diagonalCovarianceMatrix[i] = covarianceMatrix[i][i];
        }

        Arrays.sort(diagonalCovarianceMatrix);
        double pos1 = Math.floor((diagonalCovarianceMatrix.length - 1.0) / 2.0);
        double pos2 = Math.ceil((diagonalCovarianceMatrix.length - 1.0) / 2.0);

        double median;
        if (pos1 == pos2) {
            median = diagonalCovarianceMatrix[(int) pos1];
        } else {
            median = (diagonalCovarianceMatrix[(int) pos1] + diagonalCovarianceMatrix[(int) pos2]) / 2.0;
        }

        double numerator = 0;
        double denominator = 0;
        for (int i = 1; i < channelNumber; i++) {
            double tmp = 0;
            for (int j = 0; j < minProducts; j++) {
                tmp += Math.pow(
                        (Math.pow(products[i][j] / (directMonitors[i] * delayedMonitors[i]) - meanCovariance[i], 2) -
                                diagonalCovarianceMatrix[i]), 2);
            }
            tmp *= minProducts / Math.pow(minProducts - 1, 3);
            numerator += tmp;
            denominator += Math.pow(diagonalCovarianceMatrix[i] - median, 2);
        }

        return Math.max(Math.min(1, numerator / denominator), 0);
    }

    private void calculateCF(double[][] intensityBlocks, int numFrames, int index) {
        // FIXME: implement GLS button
        boolean GLS = true;

        // intensityBlocks is the array of intensity values for the two traces witch are correlated
        blockVariance = new double[channelNumber];
        blockStandardDeviation = new double[channelNumber];

        double[][] covarianceMatrix = new double[channelNumber][channelNumber];
        // the final results does not contain information about the zero lagtime kcf
        regularizedCovarianceMatrix = new double[channelNumber - 1][channelNumber - 1];

        double[] numProducts = new double[channelNumber];
        double[][] products = new double[channelNumber][numFrames];

        double[] correlationMean = new double[channelNumber];
        meanCovariance = new double[channelNumber];

        // direct and delayed monitors required for ACF normalization
        double[] directMonitors = new double[channelNumber];
        double[] delayedMonitors = new double[channelNumber];

        int numBinnedDataPoints = numFrames;
        int currentIncrement = BLOCK_LAG;
        int minProducts = (int) (numSamples[channelNumber - 1] /
                Math.pow(2, Math.max(index - Math.log(sampleTimes[channelNumber - 1]) / Math.log(2), 0)));

        // count how often the data was binned
        int binCount = 0;

        for (int i = 0; i < channelNumber; i++) {
            if (currentIncrement != sampleTimes[i]) {
                currentIncrement = sampleTimes[i];
                numBinnedDataPoints /= 2;
                binCount++;

                binData(numBinnedDataPoints, intensityBlocks);
            }

            int delay = lags[i] / currentIncrement;
            numProducts[i] = numBinnedDataPoints - delay;

            double[] monitors = calculateMonitors(numProducts[i], intensityBlocks, delay);
            directMonitors[i] = monitors[0];
            delayedMonitors[i] = monitors[1];

            double[] sumProds = calculateCorrelations(numProducts[i], intensityBlocks, directMonitors[i],
                    delayedMonitors[i], products[i], delay);

            correlationMean[i] = sumProds[0] / (numProducts[i] * directMonitors[i] * delayedMonitors[i]);

            int binTimes = index - binCount;
            // bin the data until block time is reached
            for (int j = 1; j <= binTimes; j++) {
                // for each binning the number of data point is halfed
                numProducts[i] /= 2;
                for (int k = 0; k < numProducts[i]; k++) {
                    // do the binning and divide by 2 so that the average value does not change
                    products[i][k] = (products[i][2 * k] + products[i][2 * k + 1]) / 2;
                }
            }

            // use only the minimal number of products to achieve a symmetric variance matrix
            numProducts[i] = minProducts;

            blockVariance[i] = calculateBlockVariance(products[i], directMonitors[i], delayedMonitors[i],
                    numProducts[i]);
            blockStandardDeviation[i] = Math.sqrt(blockVariance[i]);
        }

        // TODO: check if GLS is selected, if it's selected then we do the following operations
        // if GLS is selected, then calculate the regularized covariance matrix
        if (GLS) {
            calculateMeanCovariance(products, directMonitors, delayedMonitors, minProducts);
            calculateCovarianceMatrix(covarianceMatrix, products, directMonitors, delayedMonitors, minProducts);
            double varianceShrinkageWeight =
                    calculateVarianceShrinkageWeight(covarianceMatrix, products, directMonitors,
                            delayedMonitors, minProducts);
        }
    }
}
