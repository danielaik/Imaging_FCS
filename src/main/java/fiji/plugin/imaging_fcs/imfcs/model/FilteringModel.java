package fiji.plugin.imaging_fcs.imfcs.model;

import ij.ImagePlus;

/**
 * The {@code FilteringModel} class represents a threshold model used for filtering values based on
 * minimum and maximum thresholds. It can handle separate thresholds for autocorrelation functions
 * (ACFs) and cross-correlation functions (CCFs).
 */
public class FilteringModel {
    private static ImagePlus filteringBinaryImage = null;
    private static boolean acfsSameAsCCF = false;
    private final FilteringModel acfThreshold;
    private final double fieldFactor;
    private double min;
    private double max;
    private boolean active;

    /**
     * Constructs a new {@code FilteringModel} with default settings.
     *
     * @param fieldFactor The factor by which min and max values are adjusted.
     */
    public FilteringModel(double fieldFactor) {
        // Calls the constructor that initializes acfThreshold without recursion.
        this(fieldFactor, true);
    }

    /**
     * Constructs a new {@code FilteringModel} with an option to initialize the ACF threshold.
     *
     * @param fieldFactor            The factor by which min and max values are adjusted.
     * @param initializeAcfThreshold Whether to initialize the ACF threshold.
     */
    private FilteringModel(double fieldFactor, boolean initializeAcfThreshold) {
        if (initializeAcfThreshold) {
            acfThreshold = new FilteringModel(fieldFactor, false);
        } else {
            acfThreshold = null;
        }

        this.fieldFactor = fieldFactor;
        setDefault();
    }

    /**
     * Sets whether the ACF thresholds are the same as the CCF thresholds.
     *
     * @param acfsSameAsCCF {@code true} if ACF thresholds should be the same as CCF thresholds; {@code false}
     *                      otherwise.
     */
    public static void setAcfsSameAsCCF(boolean acfsSameAsCCF) {
        FilteringModel.acfsSameAsCCF = acfsSameAsCCF;
    }

    /**
     * Retrieves the binary image used for filtering.
     *
     * @return The {@code ImagePlus} object representing the filtering binary image.
     */
    public static ImagePlus getFilteringBinaryImage() {
        return FilteringModel.filteringBinaryImage;
    }

    /**
     * Sets the binary image used for filtering.
     *
     * @param filteringBinaryImage The {@code ImagePlus} object to be set as the filtering binary image.
     */
    public static void setFilteringBinaryImage(ImagePlus filteringBinaryImage) {
        FilteringModel.filteringBinaryImage = filteringBinaryImage;
    }

    /**
     * Determines if a given value should be filtered based on the threshold settings.
     *
     * @param value The value to check against the threshold.
     * @return {@code true} if the value is outside the threshold bounds and the threshold is active,
     * {@code false} otherwise.
     */
    public boolean toFilter(double value) {
        if (active) {
            return min > value || max < value;
        }

        return false;
    }

    /**
     * Resets the threshold to its default values.
     */
    public void setDefault() {
        min = Double.POSITIVE_INFINITY;
        max = Double.NEGATIVE_INFINITY;
        active = false;
        if (acfThreshold != null) {
            acfThreshold.setDefault();
        }
    }

    /**
     * Updates the minimum and maximum threshold values based on the provided parameter.
     * If the threshold is not active, the min and max values are updated to include the parameter.
     *
     * @param parameter The value used to update the min and max thresholds.
     */
    public void updateMinMax(double parameter) {
        if (!active) {
            if (parameter > max) {
                max = parameter;
            }

            if (parameter < min) {
                min = parameter;
            }
        }
    }

    public double getMin() {
        return min * fieldFactor;
    }

    public void setMin(String min) {
        this.min = Double.parseDouble(min) / fieldFactor;
    }

    public double getMax() {
        return max * fieldFactor;
    }

    public void setMax(String max) {
        this.max = Double.parseDouble(max) / fieldFactor;
    }

    public boolean getActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public void setActive(boolean active, boolean acfActive) {
        setActive(active);
        setAcfActive(acfActive);
    }

    public boolean getAcfActive() {
        return acfThreshold.getActive();
    }

    public void setAcfActive(boolean acfActive) {
        if (acfThreshold != null) {
            acfThreshold.setActive(this.active && acfActive && !acfsSameAsCCF);
        }
    }

    public FilteringModel getAcfThreshold() {
        // if we are using the same threshold for ACF1 / ACF2, then we just return the current threshold
        if (acfsSameAsCCF) {
            return this;
        }

        return acfThreshold;
    }
}
