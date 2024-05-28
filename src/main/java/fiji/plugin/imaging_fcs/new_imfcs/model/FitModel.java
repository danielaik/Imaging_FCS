package fiji.plugin.imaging_fcs.new_imfcs.model;

import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.DIFFUSION_COEFFICIENT_BASE;
import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;

public class FitModel {
    private double D, Q2, Q3, N, F2, F3, D2, D3, G, vx, vy, fTrip, tTrip, modProb1, modProb2, modProb3;
    private int fitStart, fitEnd;

    public FitModel() {
        setDefaultValues();
    }

    public void setDefaultValues() {
        D = 1 / DIFFUSION_COEFFICIENT_BASE;
        Q2 = 1;
        Q3 = 1;
        N = 1;
        F2 = 0;
        F3 = 0;
        D2 = 0;
        D3 = 0;
        vx = 0;
        vy = 0;
        G = 0;
        fTrip = 0;
        tTrip = 0;
        fitStart = 1;
        fitEnd = 0;
        modProb1 = 0;
        modProb2 = 0;
        modProb3 = 0;
    }

    public double getD() {
        return D;
    }

    public void setD(String D) {
        this.D = Double.parseDouble(D) / DIFFUSION_COEFFICIENT_BASE;
    }

    public double getDInterface() {
        return D * DIFFUSION_COEFFICIENT_BASE;
    }

    public double getN() {
        return N;
    }

    public void setN(String N) {
        this.N = Double.parseDouble(N);
    }

    public double getVx() {
        return vx;
    }

    public void setVx(String vx) {
        this.vx = Double.parseDouble(vx) / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getVxInterface() {
        return vx * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getVy() {
        return vy;
    }

    public void setVy(String vy) {
        this.vy = Double.parseDouble(vy) / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getVyInterface() {
        return vy * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getG() {
        return G;
    }

    public void setG(String G) {
        this.G = Double.parseDouble(G);
    }

    public double getF2() {
        return F2;
    }

    public void setF2(String F2) {
        this.F2 = Double.parseDouble(F2);
    }

    public double getQ2() {
        return Q2;
    }

    public void setQ2(String Q2) {
        this.Q2 = Double.parseDouble(Q2);
    }

    public double getF3() {
        return F3;
    }

    public void setF3(String F3) {
        this.F3 = Double.parseDouble(F3);
    }

    public double getD2() {
        return D2;
    }

    public void setD2(String D2) {
        this.D2 = Double.parseDouble(D2) / DIFFUSION_COEFFICIENT_BASE;
    }

    public double getD2Interface() {
        return D2 * DIFFUSION_COEFFICIENT_BASE;
    }

    public double getD3() {
        return D3;
    }

    public void setD3(String D3) {
        this.D3 = Double.parseDouble(D3) / DIFFUSION_COEFFICIENT_BASE;
    }

    public double getD3Interface() {
        return D3 * DIFFUSION_COEFFICIENT_BASE;
    }

    public double getFTrip() {
        return fTrip;
    }

    public void setFTrip(String fTrip) {
        this.fTrip = Double.parseDouble(fTrip);
    }

    public double getTTrip() {
        return tTrip;
    }

    public void setTTrip(String tTrip) {
        this.tTrip = Double.parseDouble(tTrip);
    }

    public double getTTripInterface() {
        return tTrip * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
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
        this.Q3 = Double.parseDouble(Q3);
    }
}
