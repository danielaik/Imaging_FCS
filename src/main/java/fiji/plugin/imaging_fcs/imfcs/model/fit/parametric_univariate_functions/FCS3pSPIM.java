package fiji.plugin.imaging_fcs.imfcs.model.fit.parametric_univariate_functions;

import fiji.plugin.imaging_fcs.imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.imfcs.model.PixelModel;
import org.apache.commons.math3.special.Erf;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import static fiji.plugin.imaging_fcs.imfcs.constants.Constants.REFRACTIVE_INDEX;
import static fiji.plugin.imaging_fcs.imfcs.constants.Constants.SQRT_PI;

/**
 * This class represents the 3D Single Plane Illumination Microscopy (SPIM) model for Fluorescence Correlation
 * Spectroscopy (FCS).
 * It extends the FCSFit class and provides optimized implementations for calculating the value and gradient of the
 * model
 * using memoization and precomputations for performance improvements.
 */
public class FCS3pSPIM extends FCSFit {
    // General parameters
    private static final int Z_STEPS = 80;
    private static final int LOOP_ITERATIONS = Z_STEPS * Z_STEPS;
    private static final double INVERSE_SQRT_PI = 1.0 / SQRT_PI;
    // Caches for memoization
    private final Map<Double, double[][]> zExpCache = new ConcurrentHashMap<>();
    private final Map<ComponentKey, double[]> componentCache = new ConcurrentHashMap<>();
    private double srn, NA, modifiedObservationVolume;
    // Precomputed arrays
    private double[] zValues;
    private double[] psfZValues;

    /**
     * Constructor for the FCS3pSPIM class.
     *
     * @param settings The experimental settings model.
     * @param fitModel The fit model.
     */
    public FCS3pSPIM(ExpSettingsModel settings, FitModel fitModel) {
        super(settings, fitModel, 0);
        precomputeZValues();
    }

    @Override
    protected void initParameters(ExpSettingsModel settings, int mode) {
        super.initParameters(settings, mode);

        NA = settings.getNA();
        srn = Math.sqrt(Math.pow(REFRACTIVE_INDEX, 2) - Math.pow(NA, 2));
        modifiedObservationVolume = getModifiedObservationVolume(settings);
    }

    /**
     * Precomputes zValues and psfZValues arrays.
     */
    private void precomputeZValues() {
        zValues = new double[Z_STEPS];
        psfZValues = new double[Z_STEPS];
        for (int i = 0; i < Z_STEPS; i++) {
            zValues[i] = calculateZ(i);
            psfZValues[i] = calculatePSFz(zValues[i]);
        }
    }

    /**
     * Calculates the modified observation volume.
     *
     * @param settings The experimental settings model.
     * @return The modified observation volume.
     */
    private double getModifiedObservationVolume(ExpSettingsModel settings) {
        double fitObservationVolume = getFitObservationVolume(ax, ay, s);

        // Additional calculations for 3D volume
        double psfZ = 2 * settings.getEmLambda() * REFRACTIVE_INDEX / Math.pow(NA, 2.0);
        double szEff = Math.sqrt(1 / (Math.pow(sz, -2.0) + Math.pow(psfZ, -2.0)));

        double volume3d = SQRT_PI * szEff * fitObservationVolume;
        return volume3d / (SQRT_PI * sz);
    }

    /**
     * Calculates the z coordinate based on the index.
     *
     * @param index The index.
     * @return The z coordinate.
     */
    private double calculateZ(int index) {
        return (sz * (index - 40)) / 20;
    }

    /**
     * Calculates the PSF in the z direction.
     *
     * @param z The z coordinate.
     * @return The PSF in the z direction.
     */
    private double calculatePSFz(double z) {
        return s + (NA * Math.abs(z)) / srn;
    }

    /**
     * Calculates the z exponential term.
     *
     * @param z1 The first z coordinate.
     * @param z2 The second z coordinate.
     * @param D  The diffusion coefficient.
     * @param x  The lag time.
     * @return The z exponential term.
     */
    private double calculateZExp(double z1, double z2, double D, double x) {
        double zdiff = z1 - z2;
        double z1exp = (2 / (sz * sz)) * (z1 * z1 + z2 * z2);
        double z2exp = (zdiff * zdiff) / (4 * D * x);
        return Math.exp(-(z1exp + z2exp));
    }

    /**
     * Retrieves precomputed zExpValues for a given D and x, or computes them if not already cached.
     *
     * @param D Diffusion coefficient.
     * @param x Lag time.
     * @return Precomputed or newly computed z exponential values.
     */
    private double[][] getZExpValues(double D, double x) {
        double key = D * x;
        return zExpCache.computeIfAbsent(key, k -> {
            double[][] zExpValues = new double[Z_STEPS][Z_STEPS];
            for (int i = 0; i < Z_STEPS; i++) {
                double z1 = zValues[i];
                for (int j = 0; j < Z_STEPS; j++) {
                    double z2 = zValues[j];
                    zExpValues[i][j] = calculateZExp(z1, z2, D, x);
                }
            }
            return zExpValues;
        });
    }

    /**
     * Calculates a component and its derivative.
     *
     * @param x    The lag time.
     * @param a    Parameter a.
     * @param r    Parameter r.
     * @param sdt  Standard deviation of t.
     * @param sp0t Square root of p0t.
     * @param p0t  p0t value.
     * @return An array containing the part and the derivative.
     */
    private double[] calculateComponentWithDerivative(double x, double a, double r, double sdt, double sp0t,
                                                      double p0t) {
        ComponentKey key = new ComponentKey(a, r, sp0t);

        return componentCache.computeIfAbsent(key, k -> {
            double p1 = a + r;
            double p2 = a - r;

            double p10t = p1 / sp0t;
            double p20t = p2 / sp0t;
            double p30t = r / sp0t;
            double p1exp = Math.exp(-p10t * p10t);
            double p2exp = Math.exp(-p20t * p20t);
            double p3exp = Math.exp(-p30t * p30t);
            double pexp = p1exp + p2exp - 2 * p3exp;
            double perf = p1 * Erf.erf(p10t) + p2 * Erf.erf(p20t) - 2 * r * Erf.erf(p30t);
            double part = (pexp * sp0t) / SQRT_PI + perf;

            double tsp0t = sp0t * sp0t * sp0t;
            double qp0t = p0t * p0t;

            double d1exp = 4 * x * (p1exp * p1 * p1);
            double d2exp = 4 * x * (p2exp * p2 * p2);
            double d3exp = 4 * x * (p3exp * r * r);
            double d0exp = 2 * x * pexp;
            double dexp = d1exp + d2exp - 2 * d3exp;

            double derivative = (1 / sdt) *
                    (INVERSE_SQRT_PI * ((d0exp / sp0t) - (dexp / tsp0t)) + (dexp / SQRT_PI) * (sp0t / qp0t));

            return new double[]{part, derivative};
        });
    }

    @Override
    public double value(double x, double[] params) {
        PixelModel.FitParameters p = new PixelModel.FitParameters(fitModel.fillParamsArray(params));

        double D = p.getD();
        double sdt = Math.sqrt(D * x);

        // Get precomputed zExpValues
        double[][] zExpValues = getZExpValues(D, x);

        double sum = IntStream.range(0, LOOP_ITERATIONS).parallel().mapToDouble(i -> {
            int z1Index = i / Z_STEPS;
            int z2Index = i % Z_STEPS;

            double zExp = zExpValues[z1Index][z2Index];

            double psfxz1 = psfZValues[z1Index];
            double psfxz2 = psfZValues[z2Index];

            double p0t = ((8 * D * x) + psfxz1 * psfxz1 + psfxz2 * psfxz2) / 2;
            double sp0t = Math.sqrt(p0t);

            // Compute xPart and yPart
            double[] xComp = calculateComponentWithDerivative(x, ax, rx, sdt, sp0t, p0t);
            double[] yComp = calculateComponentWithDerivative(x, ay, ry, sdt, sp0t, p0t);

            double xPart = xComp[0];
            double yPart = yComp[0];

            return ((zExp * xPart * yPart) * ((sz * sz) / 400)) / sdt;
        }).sum();

        double acf1 = (sum * 1e6) / (4 * ax * ax * ay * ay / modifiedObservationVolume);
        double triplet = 1 + p.getFTrip() / (1 - p.getFTrip()) * Math.exp(-x / p.getTTrip());

        return (acf1 / p.getN()) * triplet + p.getG();
    }

    @Override
    public double[] gradient(double x, double[] params) {
        PixelModel.FitParameters p = new PixelModel.FitParameters(fitModel.fillParamsArray(params));

        double D = p.getD();
        double sdt = Math.sqrt(D * x);

        // Get precomputed zExpValues
        double[][] zExpValues = getZExpValues(D, x);

        double[] sums = IntStream.range(0, LOOP_ITERATIONS).parallel().mapToObj(i -> {
            int z1Index = i / Z_STEPS;
            int z2Index = i % Z_STEPS;

            double zExp = zExpValues[z1Index][z2Index];

            double psfxz1 = psfZValues[z1Index];
            double psfxz2 = psfZValues[z2Index];

            double p0t = ((8 * D * x) + psfxz1 * psfxz1 + psfxz2 * psfxz2) / 2;
            double sp0t = Math.sqrt(p0t);

            // Compute xPart and yPart with derivatives
            double[] xComp = calculateComponentWithDerivative(x, ax, rx, sdt, sp0t, p0t);
            double[] yComp = calculateComponentWithDerivative(x, ay, ry, sdt, sp0t, p0t);

            double xPart = xComp[0];
            double xDerivative = xComp[1];
            double yPart = yComp[0];
            double yDerivative = yComp[1];

            double z1 = zValues[z1Index];
            double z2 = zValues[z2Index];
            double zDiff = z1 - z2;

            double dt1 = -(0.5 * x) / (sdt * sdt * sdt);
            double dt2 = (0.25 * (zDiff * zDiff)) / (x * sdt * D * D);

            double sumTerm = ((zExp * xPart * yPart) * ((sz * sz) / 400)) / sdt;
            double derivativeTerm =
                    zExp * ((dt1 + dt2) * xPart * yPart + xDerivative * yPart + xPart * yDerivative) * (sz * sz / 400);

            return new double[]{sumTerm, derivativeTerm};
        }).reduce(new double[]{0, 0}, (a, b) -> new double[]{a[0] + b[0], a[1] + b[1]});

        double sum = sums[0];
        double sumDerivative = sums[1];

        double acf1 = (sum * 1e6) / (4 * ax * ax * ay * ay / modifiedObservationVolume);
        double Dpspim = (sumDerivative * 1e6) / (4 * ax * ax * ay * ay / modifiedObservationVolume);

        // Triplet correction
        double triplet = 1 + p.getFTrip() / (1 - p.getFTrip()) * Math.exp(-x / p.getTTrip());
        double dTripletFtrip = Math.exp(-x / p.getTTrip()) *
                (1 / (1 - p.getFTrip()) + p.getFTrip() / ((1 - p.getFTrip()) * (1 - p.getFTrip())));
        double dTripletTtrip =
                Math.exp(-x / p.getTTrip()) * (p.getFTrip() * x) / ((1 - p.getFTrip()) * p.getTTrip() * p.getTTrip());

        double pacf = (acf1 / p.getN()) * triplet + p.getG();

        double[] results = new double[]{
                (-acf1 * triplet) / (p.getN() * p.getN()), (Dpspim / p.getN()), 0,
                // Vx derivative (not applicable)
                0,
                // Vy derivative (not applicable)
                1,
                // G derivative
                0,
                // F2 derivative (not applicable)
                0,
                // D2 derivative (not applicable)
                0,
                // F3 derivative (not applicable)
                0,
                // D3 derivative (not applicable)
                dTripletFtrip * pacf, dTripletTtrip * pacf
        };

        return fitModel.filterFitArray(results);
    }

    /**
     * Key class for caching components.
     */
    private static class ComponentKey {
        private final double a;
        private final double r;
        private final double sp0t;

        public ComponentKey(double a, double r, double sp0t) {
            this.a = a;
            this.r = r;
            this.sp0t = sp0t;
        }

        @Override
        public int hashCode() {
            int result = Double.hashCode(a);
            result = 31 * result + Double.hashCode(r);
            result = 31 * result + Double.hashCode(sp0t);
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof ComponentKey))
                return false;
            ComponentKey other = (ComponentKey) obj;
            return Double.compare(a, other.a) == 0 && Double.compare(r, other.r) == 0 &&
                    Double.compare(sp0t, other.sp0t) == 0;
        }
    }
}