package fiji.plugin.imaging_fcs.new_imfcs.model.fit;

import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;

import java.util.ArrayList;

/**
 * Abstract class for curve fitting of intensity traces using time data.
 */
public abstract class BaseFit extends AbstractCurveFitter {
    protected final double[] intensityTime;

    /**
     * Constructs a BaseFit with specified time data.
     *
     * @param intensityTime Time data array for intensity measurements.
     */
    public BaseFit(double[] intensityTime) {
        super();
        this.intensityTime = intensityTime;
    }

    /**
     * Fits a curve to the provided intensity trace data.
     *
     * @param intensityTrace Array of intensity data to fit.
     * @return Fitted parameters as a double array.
     */
    public double[] fitIntensityTrace(double[] intensityTrace) {
        ArrayList<WeightedObservedPoint> points = new ArrayList<>();
        int numPoints = intensityTrace.length;

        for (int i = 0; i < numPoints; i++) {
            points.add(new WeightedObservedPoint(1, intensityTime[i], intensityTrace[i]));
        }

        return fit(points);
    }
}
