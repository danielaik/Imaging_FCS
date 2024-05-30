package fiji.plugin.imaging_fcs.new_imfcs.model.fit;

import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;

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
     * Fills the target and weights arrays with the Y values and weights from the given collection of
     * {@link WeightedObservedPoint} objects.
     *
     * @param points  the collection of {@link WeightedObservedPoint} objects to process
     * @param target  the array to be filled with the Y values of the points
     * @param weights the array to be filled with the weights of the points
     * @throws IllegalArgumentException if the length of the target or weights array does not match the
     *                                  size of the points collection
     */
    protected static void fillTargetAndWeights(Collection<WeightedObservedPoint> points, double[] target,
                                               double[] weights) {
        if (points.size() != target.length || points.size() != weights.length) {
            throw new IllegalArgumentException(
                    "The length of target and weights arrays must match the size of the points collection");
        }

        AtomicInteger index = new AtomicInteger(0);
        points.forEach(point -> {
            int i = index.getAndIncrement();
            target[i] = point.getY();
            weights[i] = point.getWeight();
        });
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
