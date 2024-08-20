package fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions;

import org.apache.commons.math3.analysis.ParametricUnivariateFunction;

/**
 * Line is a simple parametric univariate function representing a straight line.
 * It is primarily used for fitting a linear relationship in a diffusion law plot.
 * The function is defined as: y = t0 + b * x
 * where t0 is the intercept and b is the slope.
 */
public class Line implements ParametricUnivariateFunction {

    /**
     * Evaluates the line function at a given point x using the specified parameters.
     *
     * @param x          The independent variable (input).
     * @param parameters The array of parameters where parameters[0] is the intercept (t0)
     *                   and parameters[1] is the slope (b).
     * @return The value of the function at the given point x.
     */
    @Override
    public double value(double x, double[] parameters) {
        double t0 = parameters[0];
        double b = parameters[1];

        return t0 + b * x;
    }

    /**
     * Computes the gradient (partial derivatives) of the line function with respect to the parameters.
     * The gradient is returned as an array where the first element is the partial derivative with respect
     * to the intercept (t0) and the second element is the partial derivative with respect to the slope (b).
     *
     * @param x          The independent variable (input).
     * @param parameters The array of parameters where parameters[0] is the intercept (t0)
     *                   and parameters[1] is the slope (b).
     * @return An array representing the gradient [∂y/∂t0, ∂y/∂b].
     */
    @Override
    public double[] gradient(double x, double[] parameters) {
        return new double[]{1, x};
    }
}
