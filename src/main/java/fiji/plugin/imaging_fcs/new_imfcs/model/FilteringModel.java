package fiji.plugin.imaging_fcs.new_imfcs.model;

import ij.ImagePlus;

public class FilteringModel {
    private static ImagePlus filteringBinaryImage = null;
    private static boolean acfsSameAsCCF = false;
    private final FilteringModel acfThreshold;
    private double min;
    private double max;
    private boolean active;

    /**
     * Constructs a new {@code Threshold} with default settings.
     */
    public FilteringModel() {
        // Calls the constructor that initializes acfThreshold without recursion.
        this(true);
    }

    private FilteringModel(boolean initializeAcfThreshold) {
        if (initializeAcfThreshold) {
            acfThreshold = new FilteringModel(false);
        } else {
            acfThreshold = null;
        }

        setDefault();
    }

    public static void setAcfsSameAsCCF(boolean acfsSameAsCCF) {
        FilteringModel.acfsSameAsCCF = acfsSameAsCCF;
    }

    public static ImagePlus getFilteringBinaryImage() {
        return FilteringModel.filteringBinaryImage;
    }

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
     * The default minimum is -0.01, the default maximum is 0.01, and the threshold is inactive.
     */
    public void setDefault() {
        min = -0.01;
        max = 0.01;
        active = false;
        if (acfThreshold != null) {
            acfThreshold.setDefault();
        }
    }

    public double getMin() {
        return min;
    }

    public void setMin(String min) {
        this.min = Double.parseDouble(min);
    }

    public double getMax() {
        return max;
    }

    public void setMax(String max) {
        this.max = Double.parseDouble(max);
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
