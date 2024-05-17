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

                int index = blockTransform(intensityBlock,
                        slidingWindowFinalFrame - slidingWindowInitialFrame + 1);
            }
        } else {
            // if sliding window is not selected, correlate the full intensity trace
            lagGroupNumber = correlatorQ;
            channelNumber = settings.getCorrelatorP() +
                    (correlatorQ - 1) * settings.getCorrelatorP() / 2 + 1;

            // TODO: check kcf (select 1, 2 or 3)
            int mode = settings.getFitModel().equals(Constants.DC_FCCS_2D) ? 2 : 1;
            double[][] intensityBlock = getIntensityBlock(img, x, y, x2, y2, initialFrame, finalFrame, mode);

            int index = blockTransform(intensityBlock, finalFrame - initialFrame + 1);
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
                for (int j = 0; j < numBinnedDataPoints; j++) {
                    intensityBlock[0][j] = intensityBlock[0][2 * j] + intensityBlock[0][2 * j + 1];
                    intensityBlock[1][j] = intensityBlock[1][2 * j] + intensityBlock[1][2 * j + 1];
                }
            }

            if (i == BLOCK_LAG) {
                calculateCorrelation(i, blockCount, numFrames, numBinnedDataPoints, intensityBlock, varianceBlocks,
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

    private void calculateCorrelation(int i, int blockCount, int numFrames, int numBinnedDataPoints,
                                      double[][] intensityBlock, double[][] varianceBlocks, double[] numProducts,
                                      int currentIncrement) {
        int delay = lags[i] / currentIncrement;
        double directMonitor = 0.0;
        double delayedMonitor = 0.0;
        numProducts[0] = numBinnedDataPoints - delay;
        for (int j = 0; j < numProducts[0]; j++) {
            directMonitor += intensityBlock[0][j];
            delayedMonitor += intensityBlock[1][j + delay];
        }
        directMonitor /= numProducts[0];
        delayedMonitor /= numProducts[0];

        double[] products = new double[numFrames];
        double sumProd = 0.0;
        double sumProdSquared = 0.0;

        // calculate the correlation
        for (int j = 0; j < numProducts[0]; j++) {
            products[j] = intensityBlock[0][j] * intensityBlock[1][j + delay] -
                    delayedMonitor * intensityBlock[0][j] - directMonitor * intensityBlock[1][j + delay] +
                    delayedMonitor * directMonitor;
            // calculate the sum of prod, i.e. the raw correlation value ...
            sumProd += products[j];
            sumProdSquared += Math.pow(products[j], 2);
        }

        varianceBlocks[0][0] = currentIncrement * settings.getFrameTime();
        varianceBlocks[1][0] = (sumProdSquared / (numBinnedDataPoints - delay) -
                Math.pow(sumProd / numProducts[0], 2)) /
                (numProducts[0] * Math.pow(directMonitor * delayedMonitor, 2));

        // perform blocking operations
        for (int j = 1; j < blockCount; j++) {
            numProducts[j] = (int) (numProducts[j - 1] / 2);
            sumProd = sumProdSquared = 0.0;
            for (int k = 0; k < numProducts[j]; k++) {
                products[k] = (products[2 * k] + products[2 * k + 1]) / 2;
                sumProd += products[k];
                sumProdSquared += products[k] * products[k];
            }

            // the time of the block curve
            varianceBlocks[0][j] = (currentIncrement * Math.pow(2, j)) * settings.getFrameTime();

            // value of the block curve
            varianceBlocks[1][j] = (sumProdSquared / numProducts[j] - Math.pow(sumProd / numProducts[j], 2)) /
                    (numProducts[j] * Math.pow(directMonitor * delayedMonitor, 2));
        }
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
}
