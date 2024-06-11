package fiji.plugin.imaging_fcs.new_imfcs.model.fit.parametric_univariate_functions;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import org.apache.commons.math3.special.Erf;

import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.*;

public class FCS_3p_SPIM extends FCSFit {
    // general parameters
    private final double INVERSE_SQRT_PI = Math.sqrt(1 / PI);
    private double p1xt, p2xt, p1yt, p2yt, srn, NA, modifiedObservationVolume;

    public FCS_3p_SPIM(ExpSettingsModel settings, FitModel fitModel) {
        super(settings, fitModel, 0);
    }

    @Override
    protected void initParameters(ExpSettingsModel settings, int mode) {
        super.initParameters(settings, mode);

        p1xt = ax + rx;
        p2xt = ax - rx;
        p1yt = ay + ry;
        p2yt = ay - ry;

        NA = settings.getNA();

        srn = Math.sqrt(Math.pow(REFRACTIVE_INDEX, 2) - Math.pow(NA, 2));

        // Calculate and store modifiedObservationVolume
        double pexpx0 = 2 * Math.exp(-Math.pow(ax / s, 2)) - 2;
        double perfx0 = 2 * ax * Erf.erf(ax / s);
        double pexpy0 = 2 * Math.exp(-Math.pow(ay / s, 2)) - 2;
        double perfy0 = 2 * ay * Erf.erf(ay / s);

        // size of PSF in axial direction convolution of two Gaussians depending on illumination profile and
        // detection PSF
        double psfZ = 2 * settings.getEmLambda() / NANO_CONVERSION_FACTOR * REFRACTIVE_INDEX / Math.pow(NA, 2.0);
        double szEff = Math.sqrt(1 / (Math.pow(sz, -2.0) + Math.pow(psfZ, -2.0)));

        double volume3d = SQRT_PI * szEff * 4 * Math.pow(ax * ay, 2) /
                ((s / SQRT_PI * pexpx0 + perfx0) * (s / SQRT_PI * pexpy0 + perfy0));
        modifiedObservationVolume = (volume3d / (SQRT_PI * sz));
    }

    @Override
    public double[] gradient(double x, double[] params) {
        double z1;
        double z2;
        double sum1;
        double sumd1;

        double N = params[0];
        double D = params[1];
        double G = params[4];
        double fTrip = params[9];
        double tTrip = params[10];

        double sdt = Math.sqrt(D * x);

        sum1 = 0;
        sumd1 = 0;

        // numerical integration for z and z' components in the ACF. Thsi is required as
        // no analytical solutions were found
        for (int i = 0; i < 6401; i++) {
            int outerloop = i / 80;
            int z1calculator = outerloop - 40;
            z1 = (sz * z1calculator) / 20;
            int z2calculator = (i % 80) - 40;

            double psfxz1 = s + (NA * Math.abs(z1)) / srn;

            z2 = (sz * z2calculator) / 20;
            double psfxz2 = s + (NA * Math.abs(z2)) / srn;

            // COMPONENT1
            // help variables, which are dependent on time, to write the full function
            double p0t = ((8 * D * x) + Math.pow(psfxz1, 2) + Math.pow(psfxz2, 2)) / 2;
            double sp0t = Math.sqrt(p0t);
            double tsp0t = Math.pow(sp0t, 3);
            double qp0t = Math.pow(p0t, 2);

            double p10xt = p1xt / sp0t;
            double p20xt = p2xt / sp0t;
            double p30xt = rx / sp0t;
            double p1expxt = Math.exp(-Math.pow(p10xt, 2));
            double p2expxt = Math.exp(-Math.pow(p20xt, 2));
            double p3expxt = Math.exp(-Math.pow(p30xt, 2));
            double pexpxt = p1expxt + p2expxt - (2 * p3expxt);
            double perfxt = (p1xt * Erf.erf(p10xt)) + (p2xt * Erf.erf(p20xt)) - (2 * rx * Erf.erf(p30xt));
            double d1expx = 4 * x * (p1expxt * Math.pow(p1xt, 2));
            double d2expx = 4 * x * (p2expxt * Math.pow(p2xt, 2));
            double d3expx = 4 * x * (p3expxt * Math.pow(rx, 2));
            double d0expx = 2 * x * (pexpxt);
            double dexpx = d1expx + d2expx - (2 * d3expx);
            double xpart = ((pexpxt * sp0t) / SQRT_PI) + perfxt;
            double xder = (1 / sdt) *
                    (INVERSE_SQRT_PI * ((d0expx / sp0t) - (dexpx / tsp0t)) + (dexpx / SQRT_PI) * (sp0t / qp0t));

            double p10yt = p1yt / sp0t;
            double p20yt = p2yt / sp0t;
            double p30yt = ry / sp0t;
            double p1expyt = Math.exp(-Math.pow(p10yt, 2));
            double p2expyt = Math.exp(-Math.pow(p20yt, 2));
            double p3expyt = Math.exp(-Math.pow(p30yt, 2));
            double pexpyt = p1expyt + p2expyt - (2 * p3expyt);
            double perfyt = (p1yt * Erf.erf(p10yt)) + (p2yt * Erf.erf(p20yt)) - (2 * ry * Erf.erf(p30yt));
            double d1expy = 4 * x * (p1expyt * Math.pow(p1yt, 2));
            double d2expy = 4 * x * (p2expyt * Math.pow(p2yt, 2));
            double d3expy = 4 * x * (p3expyt * Math.pow(ry, 2));
            double d0expy = 2 * x * (pexpyt);
            double dexpy = d1expy + d2expy - (2 * d3expy);
            double ypart = ((pexpyt * sp0t) / SQRT_PI) + perfyt;
            double yder = (1 / sdt) *
                    (INVERSE_SQRT_PI * ((d0expy / sp0t) - (dexpy / tsp0t)) + (dexpy / SQRT_PI) * (sp0t / qp0t));

            double zdiff = (z1 - z2);
            double z1exp = (2 / Math.pow(sz, 2)) * (Math.pow(z1, 2) + Math.pow(z2, 2));
            double z2exp = (Math.pow(zdiff, 2)) / (4 * D * x);
            double zexp = Math.exp(-(z1exp + z2exp));

            double dt1 = -(0.5 * x) / (Math.pow(sdt, 3));
            double dt2 = (0.25 * (Math.pow((zdiff), 2))) / (x * sdt * Math.pow(D, 2));

            // TODO: double check that the two values are correct
            sum1 += ((zexp * xpart * ypart) * ((sz * sz) / 400)) / sdt;
            sumd1 += zexp * ((dt1 + dt2) * xpart * ypart + xpart * yder + ypart * xder) * (sz * sz / 400);
        }

        double acf1 = (sum1 * 1000000) / (4 * Math.pow(ax * ay, 2) / (modifiedObservationVolume));
        double Dpspim = (sumd1 * 1000000) / (4 * Math.pow(ax * ay, 2) / (modifiedObservationVolume));
        // TRIPLET
        double triplet = 1 + fTrip / (1 - fTrip) * Math.exp(-x / tTrip);
        double dtripletFtrip = Math.exp(-x / tTrip) * (1 / (1 - fTrip) + fTrip / Math.pow(1 - fTrip, 2));
        double dtripletTtrip = Math.exp(-x / tTrip) * (fTrip * x) / ((1 - fTrip) * Math.pow(tTrip, 2));

        double pacf = ((1 / N) * acf1) * triplet + G;

        double[] grad = new double[]{
                (-1 / Math.pow(N, 2)) * acf1 * triplet,
                (1 / N) * (Dpspim), 0, 0, 1, 0, 0, 0, 0, dtripletFtrip * pacf, dtripletTtrip * pacf};

        // TODO: filter only the non hold cases
        return grad;
    }


    @Override
    public double value(double x, double[] params) {
        double N = params[0];
        double D = params[1];
        double G = params[4];
        double fTrip = params[9];
        double tTrip = params[10];

        double z1;
        double z2;
        double sum1 = 0;

        // obtain fit parameters

        double sdt = Math.sqrt(D * x);

        for (int i = 0; i < 6401; i++) {
            int outerloop = i / 80;
            int z1calculator = outerloop - 40;
            z1 = (sz * z1calculator) / 20;
            int z2calculator = (i % 80) - 40;

            double psfxz1 = s + (NA * Math.abs(z1)) / srn;

            z2 = (sz * z2calculator) / 20;
            double psfxz2 = s + (NA * Math.abs(z2)) / srn;

            // COMPONENT1
            // help variables, which are dependent on time, to write the full function
            double p0t = ((8 * D * x) + Math.pow(psfxz1, 2) + Math.pow(psfxz2, 2)) / 2;
            double sp0t = Math.sqrt(p0t);

            double p10xt = p1xt / sp0t;
            double p20xt = p2xt / sp0t;
            double p30xt = rx / sp0t;
            double p1expxt = Math.exp(-Math.pow(p10xt, 2));
            double p2expxt = Math.exp(-Math.pow(p20xt, 2));
            double p3expxt = Math.exp(-Math.pow(p30xt, 2));
            double pexpxt = p1expxt + p2expxt - (2 * p3expxt);
            double perfxt = (p1xt * Erf.erf(p10xt)) + (p2xt * Erf.erf(p20xt)) - (2 * rx * Erf.erf(p30xt));
            double xpart = ((pexpxt * sp0t) / SQRT_PI) + perfxt;

            double p10yt = p1yt / sp0t;
            double p20yt = p2yt / sp0t;
            double p30yt = ry / sp0t;
            double p1expyt = Math.exp(-Math.pow(p10yt, 2));
            double p2expyt = Math.exp(-Math.pow(p20yt, 2));
            double p3expyt = Math.exp(-Math.pow(p30yt, 2));
            double pexpyt = p1expyt + p2expyt - (2 * p3expyt);
            double perfyt = (p1yt * Erf.erf(p10yt)) + (p2yt * Erf.erf(p20yt)) - (2 * ry * Erf.erf(p30yt));
            double ypart = ((pexpyt * sp0t) / SQRT_PI) + perfyt;
            double zdiff = (z1 - z2);
            double z1exp = (2 / Math.pow(sz, 2)) * (Math.pow(z1, 2) + Math.pow(z2, 2));
            double z2exp = (Math.pow(zdiff, 2)) / (4 * D * x);
            double zexp = Math.exp(-(z1exp + z2exp));

            sum1 += ((zexp * xpart * ypart) * ((sz * sz) / 400)) / sdt;
        }

        double acf1 = (sum1 * 1000000) / (4 * Math.pow(ax * ay, 2) / (modifiedObservationVolume));
        double triplet = 1 + fTrip / (1 - fTrip) * Math.exp(-x / tTrip);

        return ((1 / N) * acf1) * triplet + G;
    }
}