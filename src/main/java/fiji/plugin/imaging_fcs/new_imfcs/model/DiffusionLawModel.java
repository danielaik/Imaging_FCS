package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.controller.InvalidUserInputException;

/**
 * The {@code DiffusionLawModel} class represents the data model for diffusion law analysis.
 */
public class DiffusionLawModel {
    private static final int MAX_POINTS = 30;
    private static final int DIMENSION_ROI = 7;

    private int binningStart = 1;
    private int binningEnd = 5;
    private int fitStart = 1;
    private int fitEnd = 5;

    /**
     * Constructs a new {@code DiffusionLawModel} object.
     * Initializes the model with default values for binning and fitting ranges.
     */
    public DiffusionLawModel() {
    }

    /**
     * Resets the binning and fitting range values to their default states.
     * This is typically called when the user switches to ROI mode.
     */
    public void resetRangeValues() {
        this.binningEnd = 5;
        this.fitStart = 1;
        this.fitEnd = 5;
    }

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
}
