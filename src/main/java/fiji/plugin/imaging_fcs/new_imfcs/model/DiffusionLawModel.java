package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.InvalidUserInputException;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions.FCSFit;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Pair;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Range;

import java.awt.*;

/**
 * The {@code DiffusionLawModel} class represents the data model for diffusion law analysis.
 */
public class DiffusionLawModel {
    private static final int MAX_POINTS = 30;
    private static final int DIMENSION_ROI = 7;
    private final ExpSettingsModel interfaceSettings;
    private final FitModel interfaceFitModel;
    private final ImageModel imageModel;
    private int binningStart = 1;
    private int binningEnd = 5;
    private int fitStart = 1;
    private int fitEnd = 5;
    private double[] observationVolumes;
    private double[] averageD;
    private double[] varianceD;
    private String mode = "All";

    /**
     * Constructs a new {@code DiffusionLawModel} object.
     * Initializes the model with provided experimental settings, image data, and fitting model.
     *
     * @param settings   the experimental settings model containing parameters for the analysis.
     * @param imageModel the image model containing the data to be analyzed.
     * @param fitModel   the fitting model used to fit the correlation data.
     */
    public DiffusionLawModel(ExpSettingsModel settings, ImageModel imageModel, FitModel fitModel) {
        // copy the settings to a new instance to be able to update the binning if needed without updating the whole
        // interface.
        this.interfaceSettings = settings;
        this.interfaceFitModel = fitModel;
        this.imageModel = imageModel;
    }

    /**
     * Initializes a {@code Correlator} object for computing correlations in the provided image data.
     *
     * @param settings   the experimental settings model used for the correlation computation.
     * @param fitModel   the fitting model containing parameters for fitting the correlation data.
     * @param imageModel the image model containing the image data to be analyzed.
     * @return a {@code Correlator} initialized with the provided settings and image data.
     */
    private Correlator initCorrelator(ExpSettingsModel settings, FitModel fitModel, ImageModel imageModel) {
        BleachCorrectionModel bleachCorrectionModel = new BleachCorrectionModel(settings, imageModel);
        bleachCorrectionModel.computeNumPointsIntensityTrace(settings.getLastFrame() - settings.getFirstFrame() + 1);
        return new Correlator(settings, bleachCorrectionModel, fitModel);
    }

    /**
     * Fits a pixel at the specified coordinates and updates the diffusion coefficient statistics
     * for the current binning setting.
     *
     * @param settings   the experimental settings model used for fitting.
     * @param fitModel   the fitting model containing parameters for fitting the pixel data.
     * @param correlator the correlator used to compute correlation data for the pixel.
     * @param pixelModel the model representing the pixel's data and fit parameters.
     * @param x          the x-coordinate of the pixel to be fitted.
     * @param y          the y-coordinate of the pixel to be fitted.
     * @param index      the index corresponding to the current binning setting.
     * @return 1 if the pixel was successfully fitted, 0 otherwise.
     */
    private int fitPixelAndAddD(ExpSettingsModel settings, FitModel fitModel, Correlator correlator,
                                PixelModel pixelModel, int x, int y, int index) {
        correlator.correlatePixelModel(pixelModel, imageModel.getImage(), x, y, x, y, settings.getFirstFrame(),
                settings.getLastFrame());
        fitModel.standardFit(pixelModel, correlator.getLagTimes());

        if (pixelModel.isFitted()) {
            averageD[index] += pixelModel.getFitParams().getD();
            varianceD[index] += Math.pow(pixelModel.getFitParams().getD(), 2);

            // Return 1 to count this element in the total elements.
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * Performs the fitting process across the specified binning range.
     * Calculates the observation volumes, average diffusion coefficients, and their variances
     * for each binning setting.
     *
     * @return a pair containing a 2D array of diffusion law data and a pair of the minimum and maximum diffusion
     * values.
     */
    public Pair<double[][], Pair<Double, Double>> fit() {
        // create a new settings model to be able to update the binning size separately from the interface.
        ExpSettingsModel settings = new ExpSettingsModel(this.interfaceSettings);

        // init a new fit model based on the interface parameters and fix the parameters
        FitModel fitModel = new FitModel(settings, interfaceFitModel);
        fitModel.setFix(true);

        Correlator correlator = initCorrelator(settings, fitModel, this.imageModel);
        PixelModel pixelModel = new PixelModel();

        observationVolumes = new double[binningEnd - binningStart + 1];
        averageD = new double[binningEnd - binningStart + 1];
        varianceD = new double[binningEnd - binningStart + 1];

        for (int currentBinning = binningStart; currentBinning <= binningEnd; currentBinning++) {
            settings.setBinning(new Point(currentBinning, currentBinning));
            settings.updateSettings();

            Range[] ranges = settings.getAllArea(imageModel.getDimension());
            Range xRange = ranges[0];
            Range yRange = ranges[1];

            int index = currentBinning - binningStart;

            observationVolumes[index] =
                    FCSFit.getFitObservationVolume(settings.getParamAx(), settings.getParamAy(), settings.getParamW()) *
                            Constants.DIFFUSION_COEFFICIENT_BASE;

            int numElements = xRange.stream()
                    .mapToInt(x -> yRange.stream()
                            .mapToInt(y -> fitPixelAndAddD(settings, fitModel, correlator, pixelModel, x, y, index))
                            .sum())
                    .sum();

            // Do the average and convert to um^2/s
            averageD[index] *= Constants.DIFFUSION_COEFFICIENT_BASE / numElements;
            // normalize the average of the square
            varianceD[index] *= Math.pow(Constants.DIFFUSION_COEFFICIENT_BASE, 2) / numElements;
            varianceD[index] = varianceD[index] - Math.pow(averageD[index], 2);
        }

        return getDiffusionLawArray(observationVolumes, averageD, varianceD);
    }

    /**
     * Constructs and returns the diffusion law array along with the minimum and maximum diffusion values.
     *
     * @param observationVolumes the array of observation volumes for each binning setting.
     * @param averageD           the array of average diffusion coefficients for each binning setting.
     * @param varianceD          the array of diffusion coefficient variances for each binning setting.
     * @return a pair containing a 2D array of diffusion law data and a pair of the minimum and maximum diffusion
     * values.
     */
    private Pair<double[][], Pair<Double, Double>> getDiffusionLawArray(double[] observationVolumes, double[] averageD,
                                                                        double[] varianceD) {
        double[][] diffusionLawArray = new double[3][observationVolumes.length];
        double minValueDiffusionLaw = Double.MAX_VALUE;
        double maxValueDiffusionLaw = -Double.MAX_VALUE;

        for (int currentBinning = binningStart; currentBinning <= binningEnd; currentBinning++) {
            int index = currentBinning - binningStart;
            diffusionLawArray[0][index] = observationVolumes[index];
            diffusionLawArray[1][index] = observationVolumes[index] / averageD[index];
            diffusionLawArray[2][index] =
                    observationVolumes[index] / Math.pow(averageD[index], 2) * Math.sqrt(varianceD[index]);

            minValueDiffusionLaw = Math.min(minValueDiffusionLaw, averageD[index] - varianceD[index]);
            maxValueDiffusionLaw = Math.max(maxValueDiffusionLaw, averageD[index] + varianceD[index]);
        }
        Pair<Double, Double> minMax = new Pair<>(minValueDiffusionLaw, maxValueDiffusionLaw);

        return new Pair<>(diffusionLawArray, minMax);
    }

    /**
     * Resets the binning and fitting range values to their default states.
     * This method is typically called when the user switches to ROI (Region of Interest) mode.
     *
     * @param reset if true, resets the range values and sets the mode to "ROI"; otherwise, sets the mode to "All".
     * @return the updated mode string, either "ROI" or "All".
     */
    public String resetRangeValues(boolean reset) {
        if (reset) {
            this.binningEnd = 5;
            this.fitStart = 1;
            this.fitEnd = 5;

            this.mode = "ROI";
        } else {
            this.mode = "All";
        }

        return this.mode;
    }

    // Getter and setter methods with input validation for binning and fitting ranges.

    public int getBinningStart() {
        return binningStart;
    }

    public void setBinningStart(String binningStart) {
        int start = Integer.parseInt(binningStart);
        if (start <= 0 || start > this.binningEnd) {
            throw new InvalidUserInputException("Binning start out of range.");
        } else if (start > this.fitStart) {
            throw new InvalidUserInputException("Fit 'Start-End' range can't be outside of binning range.");
        }
        this.binningStart = start;
    }

    public int getBinningEnd() {
        return binningEnd;
    }

    public void setBinningEnd(String binningEnd) {
        int end = Integer.parseInt(binningEnd);
        if (end >= MAX_POINTS || end < this.binningStart) {
            throw new InvalidUserInputException("Binning end out of range");
        } else if (end < this.fitEnd) {
            throw new InvalidUserInputException("Fit 'Start-End' range can't be outside of binning range.");
        }
        this.binningEnd = end;
    }

    public int getFitStart() {
        return fitStart;
    }

    public void setFitStart(String fitStart) {
        int start = Integer.parseInt(fitStart);
        if (start <= 0 || start > this.fitEnd || start < this.binningStart || start > this.binningEnd) {
            throw new InvalidUserInputException("Fit start out of range.");
        }
        this.fitStart = start;
    }

    public int getFitEnd() {
        return fitEnd;
    }

    public void setFitEnd(String fitEnd) {
        int end = Integer.parseInt(fitEnd);
        if (end < this.fitStart || end > this.binningEnd) {
            throw new InvalidUserInputException("Fit end out of range.");
        }
        this.fitEnd = end;
    }

    public int getDimensionRoi() {
        return DIMENSION_ROI;
    }

    public String getMode() {
        return mode;
    }
}
