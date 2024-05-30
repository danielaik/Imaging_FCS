package fiji.plugin.imaging_fcs.new_imfcs.model.fit;

import fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions.DoubleExponential;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.DiagonalMatrix;

import java.util.Collection;

/**
 * Implements curve fitting for intensity data using a double exponential decay model.
 */
public class DoubleExponentialFit extends BaseFit {
    /**
     * Initializes a new instance with time data for intensity measurements.
     *
     * @param intensityTime Time data array.
     */
    public DoubleExponentialFit(double[] intensityTime) {
        super(intensityTime);
    }

    /**
     * Sets up a least squares problem for fitting a double exponential model to observed data points.
     *
     * @param points Collection of weighted observed points for fitting.
     * @return Configured least squares problem suitable for optimization.
     */
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> points) {
        final int len = points.size();
        final double[] target = new double[len];
        final double[] weights = new double[len];
        final double[] initialGuess = new double[5];

        fillTargetAndWeights(points, target, weights);

        // initial guesses
        initialGuess[0] = target[1] / 2; // amplitude for first and second decay are set equal; estimated from half
        // of the first point
        initialGuess[1] = intensityTime[len / 10]; // use a tenth of the intensity trace time as first
        // estimate for the first exponential decay
        initialGuess[2] = target[1] / 2; // amplitude estimated from half of the first point
        initialGuess[3] = intensityTime[len / 2]; // use half the intensity trace time as first estimate
        // for the second exponential decay
        initialGuess[4] = 0;

        ParametricUnivariateFunction function = new DoubleExponential();

        final AbstractCurveFitter.TheoreticalValuesFunction model = new AbstractCurveFitter.TheoreticalValuesFunction(
                function, points);

        return new LeastSquaresBuilder().maxEvaluations(Integer.MAX_VALUE).maxIterations(Integer.MAX_VALUE)
                .start(initialGuess).target(target).weight(new DiagonalMatrix(weights))
                .model(model.getModelFunction(), model.getModelFunctionJacobian()).build();
    }
}
