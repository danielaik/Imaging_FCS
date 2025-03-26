package fiji.plugin.imaging_fcs.imfcs.model.fit;

import fiji.plugin.imaging_fcs.imfcs.model.fit.parametric_univariate_functions.Line;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;

import java.util.ArrayList;
import java.util.Collection;

/**
 * LineFit is a simple linear fitting class primarily used for diffusion law fitting.
 * It extends the BaseFit class and leverages the Apache Commons Math library to
 * perform a least squares fit to a straight line.
 */
public class LineFit extends BaseFit {

    /**
     * Extracts the target values, x values, and weights from a collection of
     * WeightedObservedPoint objects.
     *
     * @param points  The collection of WeightedObservedPoint objects containing the observed data.
     * @param target  The array to store the target (y) values.
     * @param xTarget The array to store the x values.
     * @param weights The array to store the weights.
     */
    private void extractDataFromPoints(Collection<WeightedObservedPoint> points, double[] target, double[] xTarget,
                                       double[] weights) {
        int i = 0;
        for (WeightedObservedPoint point : points) {
            target[i] = point.getY();
            xTarget[i] = point.getX();
            weights[i++] = point.getWeight();
        }
    }

    /**
     * Constructs a LeastSquaresProblem for fitting a line to the given collection of points.
     *
     * @param points The collection of WeightedObservedPoint objects representing the data to fit.
     * @return A LeastSquaresProblem that can be used to solve for the best-fit line parameters.
     */
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> points) {
        final int len = points.size();

        // Extract target, xTarget, and weights from points
        double[] target = new double[len];
        double[] xTarget = new double[len];
        double[] weights = new double[len];
        extractDataFromPoints(points, target, xTarget, weights);

        // initial guesses
        final double[] initialGuess = new double[2];
        // use first point as intercept estimate
        initialGuess[0] = target[0];
        // use slope calculated from first two points as slope estimate
        initialGuess[1] = (target[1] - target[0]) / (xTarget[1] - xTarget[0]);

        ParametricUnivariateFunction function = new Line();

        return getLeastSquaresProblem(points, function, initialGuess, target, weights);
    }

    /**
     * Performs a line fit on the provided data.
     *
     * @param xTrace  The array of x values (independent variable).
     * @param trace   The array of y values (dependent variable).
     * @param sdTrace The array of standard deviations for the y values.
     * @param num     The number of points to fit.
     * @return The fitted parameters (intercept and slope).
     */
    public double[] doFit(double[] xTrace, double[] trace, double[] sdTrace, int num) {
        ArrayList<WeightedObservedPoint> points = new ArrayList<>();

        // Add points here
        for (int i = 0; i < num; i++) {
            points.add(new WeightedObservedPoint(1 / sdTrace[i] / sdTrace[i], xTrace[i], trace[i]));
        }

        return this.fit(points);
    }
}
