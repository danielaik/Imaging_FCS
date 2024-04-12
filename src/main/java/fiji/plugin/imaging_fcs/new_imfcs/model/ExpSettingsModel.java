package fiji.plugin.imaging_fcs.new_imfcs.model;

import java.awt.*;

/**
 * Represents the experimental settings for imaging FCS.
 * This model includes various parameters related to imaging,
 * such as pixel size, magnification, numerical aperture (NA),
 * and others relevant for fluorescence correlation spectroscopy (FCS) analysis.
 */
public class ExpSettingsModel {
    // User parameters with default values
    //// Parameter that updates the non-user parameters
    private final Point binning = new Point(1, 1);
    private final Dimension CCF = new Dimension(0, 0);
    private double pixelSize = 24;
    private double magnification = 100;
    private double NA = 1.49;
    private double sigma = 0.8;
    private double emLambda = 515;
    private double sigma2 = 0.8;
    private double emLamdba2 = 600;
    private double sigmaZ = 1000000;
    private double sigmaZ2 = 1000000;

    //// Other user parameters
    private int firstFrame = 1;
    private int lastFrame = 1;
    private double frameTime = 0.001;
    private int correlatorP = 16;
    private int correlatorQ = 8;
    private String fitModel = "ITIR-FCS (2D)";
    private String paraCor = "N vs D";
    private String dCCF = "x direction";
    private String bleachCorrection = "none";
    private String filter = "none";
    private int filterLowerLimit = 0;
    private int filterUpperLimit = 65536;
    private boolean FCSSDisp = false;

    // Non-user parameters (compute using user parameters)
    private double paramA;
    private double paramW;
    private double paramW2;
    private double paramZ;
    private double paramZ2;
    private double paramRx;
    private double paramRy;

    /**
     * Constructs an ExpSettingsModel and initializes settings by calling updateSettings.
     */
    public ExpSettingsModel() {
        updateSettings();
    }

    /**
     * Updates the derived parameters based on the current settings.
     * This includes calculations for resolution and displacement adjustments.
     */
    public void updateSettings() {
        // Calculation of the axial resolution adjustment parameter based on pixel size and magnification.
        paramA = pixelSize * 1000 / magnification * binning.x;

        // Calculation of lateral and axial resolutions for both emission wavelengths.
        paramW = sigma * emLambda / NA;
        paramW2 = sigma2 * emLamdba2 / NA;
        paramZ = sigmaZ * emLambda / NA;
        paramZ2 = sigmaZ2 * emLamdba2 / NA;

        // Adjustments for lateral displacements, intended to reflect CCF shifts but currently using dimension directly.
        paramRx = pixelSize * 1000 / magnification * CCF.width; // FIXME: supposed to be "cfXshift"
        paramRy = pixelSize * 1000 / magnification * CCF.height; // FIXME: supposed to be "cfYshit"
    }

    // Getters and setters for various parameters follow, allowing external modification and access to the settings.
    // These include straightforward implementations to set and get values for pixel size, magnification, NA, etc.
    // Some setters parse strings to double values, enabling easy handling of text input.
    // Additionally, there are methods to get and set the binning and CCF dimensions as strings for user interfaces.
    public double getPixelSize() {
        return pixelSize;
    }

    public void setPixelSize(String pixelSize) {
        this.pixelSize = Double.parseDouble(pixelSize);
    }

    public double getMagnification() {
        return magnification;
    }

    public void setMagnification(String magnification) {
        this.magnification = Double.parseDouble(magnification);
    }

    public double getNA() {
        return NA;
    }

    public void setNA(String NA) {
        this.NA = Double.parseDouble(NA);
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(String sigma) {
        this.sigma = Double.parseDouble(sigma);
    }

    public double getEmLambda() {
        return emLambda;
    }

    public void setEmLambda(String emLambda) {
        this.emLambda = Double.parseDouble(emLambda);
    }

    public double getSigma2() {
        return sigma2;
    }

    public void setSigma2(String sigma2) {
        this.sigma2 = Double.parseDouble(sigma2);
    }

    public double getEmLamdba2() {
        return emLamdba2;
    }

    public void setEmLamdba2(String emLamdba2) {
        this.emLamdba2 = Double.parseDouble(emLamdba2);
    }

    public double getSigmaZ() {
        return sigmaZ;
    }

    public void setSigmaZ(String sigmaZ) {
        this.sigmaZ = Double.parseDouble(sigmaZ);
    }

    public double getSigmaZ2() {
        return sigmaZ2;
    }

    public void setSigmaZ2(String sigmaZ2) {
        this.sigmaZ2 = Double.parseDouble(sigmaZ2);
    }

    public double getParamA() {
        return paramA;
    }

    public double getParamW() {
        return paramW;
    }

    public double getParamZ() {
        return paramZ;
    }

    public double getParamZ2() {
        return paramZ2;
    }

    public double getParamRx() {
        return paramRx;
    }

    public double getParamRy() {
        return paramRy;
    }

    public Point getBinning() {
        return binning;
    }

    public void setBinning(String binning) {
        String[] parts = binning.replace(" ", "").split("x");
        this.binning.x = Integer.parseInt(parts[0]);
        this.binning.y = Integer.parseInt(parts[1]);
    }

    public String getBinningString() {
        return String.format("%d x %d", binning.x, binning.y);
    }

    public Dimension getCCF() {
        return CCF;
    }

    public void setCCF(String CCF) {
        String[] parts = CCF.replace(" ", "").split("x");
        this.CCF.width = Integer.parseInt(parts[0]);
        this.CCF.height = Integer.parseInt(parts[1]);
    }

    public String getCCFString() {
        return String.format("%d x %d", CCF.width, CCF.height);
    }

    public double getParamW2() {
        return paramW2;
    }

    public int getFirstFrame() {
        return firstFrame;
    }

    public void setFirstFrame(String firstFrame) {
        this.firstFrame = Integer.parseInt(firstFrame);
    }

    public int getLastFrame() {
        return lastFrame;
    }

    public void setLastFrame(String lastFrame) {
        this.lastFrame = Integer.parseInt(lastFrame);
    }

    public double getFrameTime() {
        return frameTime;
    }

    public void setFrameTime(String frameTime) {
        this.frameTime = Double.parseDouble(frameTime);
    }

    public int getCorrelatorP() {
        return correlatorP;
    }

    public void setCorrelatorP(String correlatorP) {
        this.correlatorP = Integer.parseInt(correlatorP);
    }

    public int getCorrelatorQ() {
        return correlatorQ;
    }

    public void setCorrelatorQ(String correlatorQ) {
        this.correlatorQ = Integer.parseInt(correlatorQ);
    }

    public String getFitModel() {
        return fitModel;
    }

    public void setFitModel(String fitModel) {
        this.fitModel = fitModel;
    }

    public String getParaCor() {
        return paraCor;
    }

    public void setParaCor(String paraCor) {
        this.paraCor = paraCor;
    }

    public String getdCCF() {
        return dCCF;
    }

    public void setdCCF(String dCCF) {
        this.dCCF = dCCF;
    }

    public String getBleachCorrection() {
        return bleachCorrection;
    }

    public void setBleachCorrection(String bleachCorrection) {
        this.bleachCorrection = bleachCorrection;
    }

    public String getFilter() {
        return filter;
    }

    public void setFilter(String filter) {
        this.filter = filter;
    }

    public boolean isFCSSDisp() {
        return FCSSDisp;
    }

    public void setFCSSDisp(String FCSSDisp) {
        this.FCSSDisp = Boolean.parseBoolean(FCSSDisp);
    }

    public int getFilterLowerLimit() {
        return filterLowerLimit;
    }

    public void setFilterLowerLimit(int filterLowerLimit) {
        this.filterLowerLimit = filterLowerLimit;
    }

    public int getFilterUpperLimit() {
        return filterUpperLimit;
    }

    public void setFilterUpperLimit(int filterUpperLimit) {
        this.filterUpperLimit = filterUpperLimit;
    }
}
