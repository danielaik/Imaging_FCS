package fiji.plugin.imaging_fcs.new_imfcs.model;

import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.DIFFUSION_COEFFICIENT_BASE;
import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;

/**
 * The PixelModel class represents a model for handling pixel data in fluorescence correlation spectroscopy (FCS)
 * analysis.
 * It includes methods for setting and getting various attributes related to autocorrelation function (ACF), variance,
 * standard deviation, and fitting parameters.
 */
public class PixelModel {
    private double[] acf;
    private double[] varianceAcf;
    private double[] standardDeviationAcf;
    private double[] fittedAcf;
    private double[] residuals;
    private double[] MSD;
    private double chi2 = 0;
    private boolean fitted = false;
    private int blocked, validPixel;
    private FitParameters fitParams;


    /**
     * Constructs a new PixelModel instance.
     */
    public PixelModel() {
    }

    /**
     * Adds the values of another PixelModel to this one using a sliding window approach.
     *
     * @param other The other PixelModel whose values are to be added.
     */
    public void addPixelModelSlidingWindow(PixelModel other) {
        if (acf == null || standardDeviationAcf == null || varianceAcf == null) {
            acf = other.acf;
            varianceAcf = other.varianceAcf;
            standardDeviationAcf = other.standardDeviationAcf;
        } else {
            for (int i = 0; i < acf.length; i++) {
                acf[i] += other.acf[i];
                varianceAcf[i] += other.varianceAcf[i];
                standardDeviationAcf[i] += other.standardDeviationAcf[i];
            }
        }
    }

    /**
     * Averages the values in the sliding window over a given number of windows.
     *
     * @param numSlidingWindow The number of sliding windows to average over.
     */
    public void averageSlidingWindow(int numSlidingWindow) {
        for (int i = 0; i < acf.length; i++) {
            acf[i] /= numSlidingWindow;
            varianceAcf[i] /= numSlidingWindow;
            standardDeviationAcf[i] /= numSlidingWindow;
        }
    }

    public double[] getAcf() {
        return acf;
    }

    public void setAcf(double[] acf) {
        this.acf = acf;
    }

    public double[] getStandardDeviationAcf() {
        return standardDeviationAcf;
    }

    public void setStandardDeviationAcf(double[] standardDeviationAcf) {
        this.standardDeviationAcf = standardDeviationAcf;
    }

    public double[] getFittedAcf() {
        return fittedAcf;
    }

    public void setFittedAcf(double[] fittedAcf) {
        this.fittedAcf = fittedAcf;
    }

    public double[] getResiduals() {
        return residuals;
    }

    public void setResiduals(double[] residuals) {
        this.residuals = residuals;
    }

    public double[] getMSD() {
        return MSD;
    }

    public void setMSD(double[] MSD) {
        this.MSD = MSD;
    }

    public FitParameters getFitParams() {
        return fitParams;
    }

    public void setFitParams(FitParameters fitParams) {
        this.fitParams = fitParams;
    }

    public double[] getVarianceAcf() {
        return varianceAcf;
    }

    public void setVarianceAcf(double[] varianceAcf) {
        this.varianceAcf = varianceAcf;
    }

    public double getChi2() {
        return chi2;
    }

    public void setChi2(double chi2) {
        this.chi2 = chi2;
    }

    public boolean isFitted() {
        return fitted;
    }

    public void setFitted(boolean fitted) {
        this.fitted = fitted;
    }

    /**
     * The FitParameters class encapsulates the fitting parameters for a pixel.
     */
    public static class FitParameters {
        private final double N, D, vx, vy, G, F2, D2, F3, D3, fTrip, tTrip;

        /**
         * Constructs a new FitParameters instance with the given parameters and fit model.
         *
         * @param params   The array of parameter values.
         * @param fitModel The fit model.
         */
        public FitParameters(double[] params, FitModel fitModel) {
            N = selectValue(fitModel.getN(), params[0]);
            D = selectValue(fitModel.getD(), params[1]);
            vx = selectValue(fitModel.getVx(), params[2]);
            vy = selectValue(fitModel.getVy(), params[3]);
            G = selectValue(fitModel.getG(), params[4]);
            F2 = selectValue(fitModel.getF2(), params[5]);
            D2 = selectValue(fitModel.getD2(), params[6]);
            F3 = selectValue(fitModel.getF3(), params[7]);
            D3 = selectValue(fitModel.getD3(), params[8]);
            fTrip = selectValue(fitModel.getFTrip(), params[9]);
            tTrip = selectValue(fitModel.getTTrip(), params[10]);
        }

        /**
         * Selects the value of the parameter, preferring the held value from the fit model if it exists.
         *
         * @param parameter The parameter from the fit model.
         * @param value     The value to select.
         * @return The selected value.
         */
        private double selectValue(FitModel.Parameter parameter, double value) {
            return parameter.isHeld() ? parameter.getValue() : value;
        }

        public double getN() {
            return N;
        }

        public double getD() {
            return D;
        }

        public double getDInterface() {
            return D * DIFFUSION_COEFFICIENT_BASE;
        }

        public double getVx() {
            return vx;
        }

        public double getVxInterface() {
            return vx * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
        }

        public double getVy() {
            return vy;
        }

        public double getVyInterface() {
            return vy * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
        }

        public double getG() {
            return G;
        }

        public double getF2() {
            return F2;
        }

        public double getD2() {
            return D2;
        }

        public double getD2Interface() {
            return D2 * DIFFUSION_COEFFICIENT_BASE;
        }

        public double getF3() {
            return F3;
        }

        public double getD3() {
            return D3;
        }

        public double getD3Interface() {
            return D3 * DIFFUSION_COEFFICIENT_BASE;
        }

        public double getFTrip() {
            return fTrip;
        }

        public double getTTrip() {
            return tTrip;
        }

        public double getTTripInterface() {
            return tTrip * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
        }
    }
}
