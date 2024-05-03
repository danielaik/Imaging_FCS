package fiji.plugin.imaging_fcs.new_imfcs.model.fit;

import fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions.SingleExponential;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.DiagonalMatrix;

import java.util.Collection;

/**
 * Provides methods to fit single exponential decay models to intensity data over time.
 */
public class SingleExponentialFit extends BaseFit {
    /**
     * Constructs a SingleExponentialFit with given time data for intensity measurements.
     *
     * @param intensityTime Time data array.
     */
    public SingleExponentialFit(double[] intensityTime) {
        super(intensityTime);
    }

    /**
     * Configures the least squares problem for fitting a single exponential decay model.
     *
     * @param points Collection of weighted observed points for fitting.
     * @return Configured least squares problem for the optimizer.
     */
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> points) {
        final int numPoints = points.size();
        final double[] target = new double[numPoints];
        final double[] weights = new double[numPoints];
        final double[] initialGuess = new double[3];

        int i = 0;
        for (WeightedObservedPoint point : points) {
            target[i] = point.getY();
            weights[i] = point.getWeight();
            i++;
        }

        // initial guesses
        initialGuess[0] = target[1]; // use first point as intercept estimate
        initialGuess[1] = intensityTime[numPoints / 2]; // use half the intensity trace time as first estimate
        // for the exponential decay
        initialGuess[2] = 0;

        ParametricUnivariateFunction function = new SingleExponential();

        final AbstractCurveFitter.TheoreticalValuesFunction model = new AbstractCurveFitter.TheoreticalValuesFunction(
                function, points);

        return new LeastSquaresBuilder().maxEvaluations(Integer.MAX_VALUE).maxIterations(Integer.MAX_VALUE)
                .start(initialGuess).target(target).weight(new DiagonalMatrix(weights))
                .model(model.getModelFunction(), model.getModelFunctionJacobian()).build();
    }
}
