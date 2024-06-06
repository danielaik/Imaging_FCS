package fiji.plugin.imaging_fcs.new_imfcs.model;

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


    public PixelModel() {

    }

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

    public static class FitParameters {
        private final double N, D, vx, vy, G, F2, D2, F3, D3, fTrip, tTrip;

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

        private double selectValue(FitModel.Parameter parameter, double value) {
            return parameter.isHeld() ? parameter.getValue() : value;
        }

        public double getN() {
            return N;
        }

        public double getD() {
            return D;
        }

        public double getVx() {
            return vx;
        }

        public double getVy() {
            return vy;
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

        public double getF3() {
            return F3;
        }

        public double getD3() {
            return D3;
        }

        public double getFTrip() {
            return fTrip;
        }

        public double getTTrip() {
            return tTrip;
        }
    }
}
