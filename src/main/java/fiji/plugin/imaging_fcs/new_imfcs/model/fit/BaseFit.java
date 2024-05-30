package fiji.plugin.imaging_fcs.new_imfcs.model.fit;

import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.DiagonalMatrix;

import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Abstract class for curve fitting.
 */
public abstract class BaseFit extends AbstractCurveFitter {
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
    protected void fillTargetAndWeights(Collection<WeightedObservedPoint> points, double[] target,
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
     * Configures the least squares problem for curve fitting.
     *
     * @param points       Collection of weighted observed points for fitting.
     * @param function     The parametric univariate function to fit.
     * @param initialGuess The initial guess array for the fitting parameters.
     * @param target       The target array of observed values.
     * @param weights      The weights array.
     * @return Configured least squares problem for the optimizer.
     */
    protected LeastSquaresProblem getLeastSquaresProblem(Collection<WeightedObservedPoint> points,
                                                         ParametricUnivariateFunction function,
                                                         double[] initialGuess,
                                                         double[] target,
                                                         double[] weights) {
        final AbstractCurveFitter.TheoreticalValuesFunction model = new AbstractCurveFitter.TheoreticalValuesFunction(
                function, points);

        return new LeastSquaresBuilder()
                .maxEvaluations(Integer.MAX_VALUE)
                .maxIterations(Integer.MAX_VALUE)
                .start(initialGuess)
                .target(target)
                .weight(new DiagonalMatrix(weights))
                .model(model.getModelFunction(), model.getModelFunctionJacobian())
                .build();
    }
}
