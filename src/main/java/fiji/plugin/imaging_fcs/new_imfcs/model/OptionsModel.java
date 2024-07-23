package fiji.plugin.imaging_fcs.new_imfcs.model;

import java.util.HashMap;
import java.util.Map;

/**
 * The OptionsModel class represents the configuration options for imaging FCS analysis.
 * It stores user preferences for plotting various curves and histograms, as well as
 * the option to use GPU acceleration if CUDA is available.
 */
public final class OptionsModel {
    private final boolean isCuda;
    private boolean plotACFCurves = true;
    private boolean plotSDCurves = true;
    private boolean plotIntensityCurves = true;
    private boolean plotResCurves = true;
    private boolean plotParaHist = true;
    private boolean plotBlockingCurve = false;
    private boolean plotCovMats = false;
    private boolean useGpu;

    /**
     * Constructs an OptionsModel with CUDA availability.
     * Initializes plotting options to default values and sets GPU usage based on CUDA availability.
     *
     * @param isCuda Indicates whether CUDA is available for GPU acceleration.
     */
    public OptionsModel(boolean isCuda) {
        this.isCuda = isCuda;

        // By default, we use Cuda if a GPU is detected
        this.useGpu = isCuda;
    }

    /**
     * Converts the current plot options to a map.
     * The keys are strings representing the plot option names, and the values are the option values.
     *
     * @return a map containing the plot options.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> data = new HashMap<>();
        data.put("Plot ACF Curves", plotACFCurves);
        data.put("Plot SD Curves", plotSDCurves);
        data.put("Plot Intensity Curves", plotIntensityCurves);
        data.put("Plot Residual Curves", plotResCurves);
        data.put("Plot Parameter Histogram", plotParaHist);
        data.put("Plot Blocking Curves", plotBlockingCurve);
        data.put("Plot Covariance Matrix", plotCovMats);

        return data;
    }

    /**
     * Loads plot options from a map.
     * The map keys should correspond to the option names used in the toMap method,
     * and the values should be the option values to set.
     *
     * @param data a map containing the plot options to be loaded.
     */
    public void fromMap(Map<String, Object> data) {
        plotACFCurves = (boolean) data.get("Plot ACF Curves");
        plotSDCurves = (boolean) data.get("Plot SD Curves");
        plotIntensityCurves = (boolean) data.get("Plot Intensity Curves");
        plotResCurves = (boolean) data.get("Plot Residual Curves");
        plotParaHist = (boolean) data.get("Plot Parameter Histogram");
        plotBlockingCurve = (boolean) data.get("Plot Blocking Curves");
        plotCovMats = (boolean) data.get("Plot Covariance Matrix");
    }

    // Getters and setters follows, setUseGpu is the only one with a specific behavior
    public boolean isPlotACFCurves() {
        return plotACFCurves;
    }

    public void setPlotACFCurves(boolean plotACFCurves) {
        this.plotACFCurves = plotACFCurves;
    }

    public boolean isPlotSDCurves() {
        return plotSDCurves;
    }

    public void setPlotSDCurves(boolean plotSDCurves) {
        this.plotSDCurves = plotSDCurves;
    }

    public boolean isPlotIntensityCurves() {
        return plotIntensityCurves;
    }

    public void setPlotIntensityCurves(boolean plotIntensityCurves) {
        this.plotIntensityCurves = plotIntensityCurves;
    }

    public boolean isPlotResCurves() {
        return plotResCurves;
    }

    public void setPlotResCurves(boolean plotResCurves) {
        this.plotResCurves = plotResCurves;
    }

    public boolean isPlotParaHist() {
        return plotParaHist;
    }

    public void setPlotParaHist(boolean plotParaHist) {
        this.plotParaHist = plotParaHist;
    }

    public boolean isPlotBlockingCurve() {
        return plotBlockingCurve;
    }

    public void setPlotBlockingCurve(boolean plotBlockingCurve) {
        this.plotBlockingCurve = plotBlockingCurve;
    }

    public boolean isPlotCovMats() {
        return plotCovMats;
    }

    public void setPlotCovMats(boolean plotCovMats) {
        this.plotCovMats = plotCovMats;
    }

    public boolean isUseGpu() {
        return useGpu;
    }

    /**
     * Sets whether to use GPU for computations, considering CUDA availability.
     * The GPU will only be used if CUDA was detected at initialization.
     *
     * @param useGpu true to enable GPU usage, false to disable.
     */
    public void setUseGpu(boolean useGpu) {
        // useGpu can only be set to true if Cuda was detected
        this.useGpu = useGpu && isCuda;
    }

    public boolean isCuda() {
        return isCuda;
    }
}
