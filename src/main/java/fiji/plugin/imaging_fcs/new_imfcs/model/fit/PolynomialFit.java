package fiji.plugin.imaging_fcs.new_imfcs.model.fit;

import fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions.Polynomial;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.DiagonalMatrix;

import java.util.Collection;

/**
 * Implements curve fitting for intensity data using a polynomial model of a specified order.
 */
public class PolynomialFit extends BaseFit {
    private final int polynomialOrder;

    /**
     * Initializes a new instance for polynomial curve fitting with specified time data and polynomial order.
     *
     * @param intensityTime   Time data array.
     * @param polynomialOrder The order of the polynomial used for fitting.
     */
    public PolynomialFit(double[] intensityTime, int polynomialOrder) {
        super(intensityTime);
        this.polynomialOrder = polynomialOrder;
    }

    /**
     * Sets up a least squares problem for fitting a polynomial model to the observed data points.
     *
     * @param points Collection of weighted observed points for fitting.
     * @return Configured least squares problem suitable for optimization.
     */
    @Override
    protected LeastSquaresProblem getProblem(Collection<WeightedObservedPoint> points) {
        final int len = points.size();
        final double[] target = new double[len];
        final double[] weights = new double[len];
        final double[] initialGuess = new double[polynomialOrder + 1];

        int i = 0;
        for (WeightedObservedPoint point : points) {
            target[i] = point.getY();
            weights[i] = point.getWeight();
            i += 1;
        }

        // initial guesses
        initialGuess[0] = target[len - 1]; // use the last point as offset estimate
        for (int j = 1; j <= polynomialOrder; j++) { // use a straight line as the first estimate
            initialGuess[j] = 0;
        }

        ParametricUnivariateFunction function = new Polynomial(polynomialOrder);

        final AbstractCurveFitter.TheoreticalValuesFunction model = new AbstractCurveFitter.TheoreticalValuesFunction(
                function, points);

        return new LeastSquaresBuilder().maxEvaluations(Integer.MAX_VALUE).maxIterations(Integer.MAX_VALUE)
                .start(initialGuess).target(target).weight(new DiagonalMatrix(weights))
                .model(model.getModelFunction(), model.getModelFunctionJacobian()).build();
    }
}
