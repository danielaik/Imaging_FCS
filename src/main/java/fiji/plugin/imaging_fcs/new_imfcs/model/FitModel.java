package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.controller.InvalidUserInputException;

import java.util.Arrays;

import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.DIFFUSION_COEFFICIENT_BASE;
import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;

public class FitModel {
    private Parameter D, N, F2, F3, D2, D3, G, vx, vy, fTrip, tTrip;
    private double modProb1, modProb2, modProb3, Q2, Q3;
    private int fitStart, fitEnd;
    private boolean fix = false;
    private boolean GLS = false;
    private boolean bayes = false;

    public FitModel() {
        initValues();
    }

    private void initValues() {
        D = new Parameter(1 / DIFFUSION_COEFFICIENT_BASE, false);
        N = new Parameter(1, false);
        F2 = new Parameter(0, true);
        F3 = new Parameter(0, true);
        D2 = new Parameter(0, true);
        D3 = new Parameter(0, true);
        vx = new Parameter(0, true);
        vy = new Parameter(0, true);
        G = new Parameter(0, false);
        fTrip = new Parameter(0, true);
        tTrip = new Parameter(0, true);

        Q2 = 1;
        Q3 = 1;
        modProb1 = 0;
        modProb2 = 0;
        modProb3 = 0;

        fitStart = 1;
        fitEnd = 0;
    }

    public void setDefaultValues() {
        D.value = 1 / DIFFUSION_COEFFICIENT_BASE;
        N.value = 1;
        F2.value = 0;
        F3.value = 0;
        D2.value = 0;
        D3.value = 0;
        vx.value = 0;
        vy.value = 0;
        G.value = 0;
        fTrip.value = 0;
        tTrip.value = 0;

        Q2 = 1;
        Q3 = 1;
        modProb1 = 0;
        modProb2 = 0;
        modProb3 = 0;

        fitStart = 1;
        fitEnd = 0;
    }

    public double[] getNonHeldParameterValues() {
        Parameter[] parameters = {N, D, vx, vy, G, F2, D2, F3, D3, fTrip, tTrip};

        return Arrays.stream(parameters)
                .filter(parameter -> !parameter.isHeld())
                .mapToDouble(Parameter::getValue)
                .toArray();
    }

    public double[] filterFitArray(double[] fitParams) {
        Parameter[] parameters = {N, D, vx, vy, G, F2, D2, F3, D3, fTrip, tTrip};

        return Arrays.stream(parameters)
                .filter(parameter -> !parameter.isHeld())
                .mapToDouble(parameter -> fitParams[Arrays.asList(parameters).indexOf(parameter)])
                .toArray();
    }

    public void updateParameterValues(PixelModel.FitParameters parameters) {
        D.value = parameters.getD();
        N.value = parameters.getN();
        F2.value = parameters.getF2();
        F3.value = parameters.getF3();
        D2.value = parameters.getD2();
        D3.value = parameters.getD3();
        vx.value = parameters.getVx();
        vy.value = parameters.getVy();
        G.value = parameters.getG();
        fTrip.value = parameters.getFTrip();
        tTrip.value = parameters.getTTrip();
    }

    public Parameter getD() {
        return D;
    }

    public void setD(String D) {
        double newValue = Double.parseDouble(D);
        if (newValue < 0) {
            throw new InvalidUserInputException("D must be positive.");
        }
        this.D.value = newValue / DIFFUSION_COEFFICIENT_BASE;
    }

    public double getDInterface() {
        return D.value * DIFFUSION_COEFFICIENT_BASE;
    }

    public Parameter getN() {
        return N;
    }

    public void setN(String N) {
        this.N.value = Double.parseDouble(N);
    }

    public Parameter getVx() {
        return vx;
    }

    public void setVx(String vx) {
        this.vx.value = Double.parseDouble(vx) / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getVxInterface() {
        return vx.value * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public Parameter getVy() {
        return vy;
    }

    public void setVy(String vy) {
        this.vy.value = Double.parseDouble(vy) / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getVyInterface() {
        return vy.value * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public Parameter getG() {
        return G;
    }

    public void setG(String G) {
        this.G.value = Double.parseDouble(G);
    }

    public Parameter getF2() {
        return F2;
    }

    public void setF2(String F2) {
        double newValue = Double.parseDouble(F2);
        if (newValue < 0) {
            throw new InvalidUserInputException("F2 must be positive.");
        }
        this.F2.value = newValue;
    }

    public double getQ2() {
        return Q2;
    }

    public void setQ2(String Q2) {
        double newValue = Double.parseDouble(Q2);
        if (newValue < 0) {
            throw new InvalidUserInputException("Q2 must be superior to 0");
        }
        this.Q2 = newValue;
    }

    public Parameter getF3() {
        return F3;
    }

    public void setF3(String F3) {
        double newValue = Double.parseDouble(F3);
        if (newValue < 0) {
            throw new InvalidUserInputException("F3 must be positive.");
        }
        this.F3.value = newValue;
    }

    public Parameter getD2() {
        return D2;
    }

    public void setD2(String D2) {
        double newValue = Double.parseDouble(D2);
        if (newValue < 0) {
            throw new InvalidUserInputException("D2 must be positive.");
        }
        this.D2.value = newValue / DIFFUSION_COEFFICIENT_BASE;
    }

    public double getD2Interface() {
        return D2.value * DIFFUSION_COEFFICIENT_BASE;
    }

    public Parameter getD3() {
        return D3;
    }

    public void setD3(String D3) {
        double newValue = Double.parseDouble(D3);
        if (newValue < 0) {
            throw new InvalidUserInputException("D3 must be positive.");
        }
        this.D3.value = newValue / DIFFUSION_COEFFICIENT_BASE;
    }

    public double getD3Interface() {
        return D3.value * DIFFUSION_COEFFICIENT_BASE;
    }

    public Parameter getFTrip() {
        return fTrip;
    }

    public void setFTrip(String fTrip) {
        this.fTrip.value = Double.parseDouble(fTrip);
    }

    public Parameter getTTrip() {
        return tTrip;
    }

    public void setTTrip(String tTrip) {
        this.tTrip.value = Double.parseDouble(tTrip);
    }

    public double getTTripInterface() {
        return tTrip.value * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getModProb1() {
        return modProb1;
    }

    public void setModProb1(String modProb1) {
        this.modProb1 = Double.parseDouble(modProb1);
    }

    public double getModProb2() {
        return modProb2;
    }

    public void setModProb2(String modProb2) {
        this.modProb2 = Double.parseDouble(modProb2);
    }

    public double getModProb3() {
        return modProb3;
    }

    public void setModProb3(String modProb3) {
        this.modProb3 = Double.parseDouble(modProb3);
    }

    public int getFitStart() {
        return fitStart;
    }

    public void setFitStart(String fitStart) {
        this.fitStart = Integer.parseInt(fitStart);
    }

    public int getFitEnd() {
        return fitEnd;
    }

    public void setFitEnd(String fitEnd) {
        this.fitEnd = Integer.parseInt(fitEnd);
    }

    public double getQ3() {
        return Q3;
    }

    public void setQ3(String Q3) {
        double newValue = Double.parseDouble(Q3);
        if (newValue < 0) {
            throw new InvalidUserInputException("Q3 must be superior to 0");
        }
        this.Q3 = newValue;
    }

    public boolean isFix() {
        return fix;
    }

    public void setFix(boolean fix) {
        this.fix = fix;
    }

    public boolean isGLS() {
        return GLS;
    }

    public void setGLS(boolean GLS) {
        this.GLS = GLS;
    }

    public boolean isBayes() {
        return bayes;
    }

    public void setBayes(boolean bayes) {
        this.bayes = bayes;
    }

    public static class Parameter {
        private double value;
        private boolean hold;

        public Parameter(double value, boolean hold) {
            this.value = value;
            this.hold = hold;
        }

        public double getValue() {
            return value;
        }

        public void setValue(double value) {
            this.value = value;
        }

        public boolean isHeld() {
            return hold;
        }

        public void setHold(boolean hold) {
            this.hold = hold;
        }

        @Override
        public String toString() {
            return String.valueOf(value);
        }
    }
}
