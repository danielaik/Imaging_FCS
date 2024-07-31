package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.InvalidUserInputException;
import ij.ImagePlus;

import java.awt.*;
import java.util.LinkedHashMap;
import java.util.Map;

import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.NANO_CONVERSION_FACTOR;
import static fiji.plugin.imaging_fcs.new_imfcs.constants.Constants.PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;

/**
 * Represents the experimental settings for imaging FCS.
 * This model includes various parameters related to imaging,
 * such as pixel size, magnification, numerical aperture (NA),
 * and others relevant for fluorescence correlation spectroscopy (FCS) analysis.
 */
public final class ExpSettingsModel {
    // User parameters with default values

    // minimum number of frames required for the sliding windows; this is used to calculate a useful correlatorq
    private final int SLIDING_WINDOW_MIN_FRAME = 20;

    //// Parameter that updates the non-user parameters
    private final Point binning = new Point(1, 1);
    private final Dimension CCF = new Dimension(0, 0);
    private double pixelSize = 24 / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    private double magnification = 100;
    private double NA = 1.49;
    private double sigma = 0.8;
    private double emLambda = 515 / NANO_CONVERSION_FACTOR;
    private double sigma2 = 0.8;
    private double emLambda2 = 600 / NANO_CONVERSION_FACTOR;
    private double sigmaZ = 1000000;
    private double sigmaZ2 = 1000000;

    //// Other user parameters
    private int firstFrame = 1;
    private int lastFrame = 0;
    private double frameTime = 0.001;
    private int correlatorP = 16;
    private int correlatorQ = 8;
    private String fitModel = Constants.ITIR_FCS_2D;
    private String paraCor = "N vs D";
    private String dCCF = "x direction";
    private String bleachCorrection = "none";
    private String filter = "none";
    private int filterLowerLimit = 0;
    private int filterUpperLimit = 65536;
    private boolean FCCSDisp = false;
    private boolean overlap = false;
    private boolean MSD3d = false;
    private boolean MSD = false;
    private int slidingWindowLength = 0;
    private int channelNumber = 0;
    private int lagGroupNumber = 0;

    // Non-user parameters (compute using user parameters)
    private double paramAx;
    private double paramAy;
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
        updateChannelNumber();
    }

    /**
     * Converts the current settings to a map.
     * The keys are strings representing the parameter names, and the values are the parameter values.
     *
     * @return a map containing the settings.
     */
    public Map<String, Object> toMapConfig() {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("Binning", getBinningString());
        data.put("CCF", getCCFString());
        data.put("Pixel size", getPixelSizeInterface());
        data.put("Magnification", getMagnification());
        data.put("NA", getNA());
        data.put("Sigma", getSigma());
        data.put("Sigma 2", getSigma2());
        data.put("Sigma Z", getSigmaZ());
        data.put("Sigma Z 2", getSigmaZ2());
        data.put("Lambda", getEmLambdaInterface());
        data.put("Lambda 2", getEmLambda2Interface());
        data.put("Frame time", getFrameTime());

        return data;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> data = toMapConfig();

        data.put("First frame", getFirstFrame());
        data.put("Last frame", getLastFrame());
        data.put("Correlator P", getCorrelatorP());
        data.put("Correlator Q", getCorrelatorQ());
        data.put("Fit model", getFitModel());
        data.put("FCCS Display", isFCCSDisp());
        data.put("Overlap", isOverlap());
        data.put("Bleach correction", getBleachCorrection());
        data.put("Sliding window length", getSlidingWindowLength());
        data.put("Filter", getFilter());
        data.put("Filter Lower Limit", getFilterLowerLimit());
        data.put("Filter Upper Limit", getFilterUpperLimit());

        return data;
    }

    /**
     * Loads settings from a map.
     * The map keys should correspond to the parameter names used in the toMap method,
     * and the values should be the parameter values to set.
     *
     * @param data a map containing the settings to be loaded.
     */
    public void fromMap(Map<String, Object> data) {
        setBinning(data.get("Binning").toString());
        setCCF(data.get("CCF").toString());
        setPixelSize(data.get("Pixel size").toString());
        setMagnification(data.get("Magnification").toString());
        setNA(data.get("NA").toString());
        setSigma(data.get("Sigma").toString());
        setSigma2(data.get("Sigma 2").toString());
        setSigmaZ(data.get("Sigma Z").toString());
        setSigmaZ2(data.get("Sigma Z 2").toString());
        setEmLambda(data.get("Lambda").toString());
        setEmLambda2(data.get("Lambda 2").toString());
        setFrameTime(data.get("Frame time").toString());

        // Update the settings accordingly.
        updateSettings();
        updateChannelNumber();
    }


    /**
     * Updates the derived parameters based on the current settings.
     * This includes calculations for resolution and displacement adjustments.
     */
    public void updateSettings() {
        // Calculation of the axial resolution adjustment parameter based on pixel size and magnification.
        paramAx = pixelSize / magnification * binning.x;
        paramAy = pixelSize / magnification * binning.y;

        // Calculation of lateral and axial resolutions for both emission wavelengths.
        paramW = sigma * emLambda / NA;
        paramW2 = sigma2 * emLambda2 / NA;
        paramZ = sigmaZ * emLambda / NA;
        paramZ2 = sigmaZ2 * emLambda2 / NA;

        // Adjustments for lateral displacements.
        int cfXShift = 0;
        int cfYShift = 0;
        if (fitModel.equals(Constants.ITIR_FCS_2D) || fitModel.equals(Constants.SPIM_FCS_3D)) {
            cfXShift = CCF.width;
            cfYShift = CCF.height;
        }

        paramRx = pixelSize / magnification * cfXShift;
        paramRy = pixelSize / magnification * cfYShift;
    }

    /**
     * Updates the channel number and lag group number based on the bleach correction method.
     */
    public void updateChannelNumber() {
        if (bleachCorrection.equals(Constants.BLEACH_CORRECTION_SLIDING_WINDOW)) {
            lagGroupNumber = (int) Math.floor(
                    (Math.log((double) slidingWindowLength / (SLIDING_WINDOW_MIN_FRAME + correlatorP)) + 1) /
                            Math.log(2));
            channelNumber = correlatorP + (lagGroupNumber - 1) * correlatorP / 2 + 1;
        } else {
            lagGroupNumber = correlatorQ;
            channelNumber = correlatorP + (correlatorQ - 1) * correlatorP / 2 + 1;
        }
    }

    /**
     * Determines the number of pixels that can be correlated, depending on whether overlap is allowed
     *
     * @param img the img to get the shape from
     * @return the dimension of the useful area
     */
    public Dimension getUsefulArea(ImagePlus img) {
        if (overlap) {
            return new Dimension(img.getWidth() - binning.x, img.getHeight() - binning.y);
        } else {
            return new Dimension((img.getWidth() / binning.x) - 1, (img.getHeight() / binning.y) - 1);
        }
    }

    /**
     * Calculates the minimum cursor position based on the given distance and pixel binning factor.
     *
     * @param distance     the distance between the two pixels
     * @param pixelBinning the pixel binning factor
     * @return the calculated minimum cursor position
     */
    private int calculateMinCursorPosition(int distance, int pixelBinning) {
        if (distance < 0) {
            return -(int) Math.floor((double) distance / pixelBinning);
        }
        return 0;
    }

    /**
     * Calculates the minimum cursor position in the image.
     *
     * @return the calculated minimum cursor position
     */
    public Point getMinCursorPosition() {
        Point pixelBinning = getPixelBinning();

        return new Point(calculateMinCursorPosition(CCF.width, pixelBinning.x),
                calculateMinCursorPosition(CCF.height, pixelBinning.y));
    }

    /**
     * Calculates the maximum cursor position based on given parameters.
     *
     * @param pixelDimension the pixel dimension
     * @param distance       the distance to be calculated
     * @param imageDimension the image dimension
     * @param pixelBinning   the pixel binning factor
     * @param binning        the binning factor
     * @return the calculated maximum cursor position
     */
    private int calculateMaxCursorPosition(int pixelDimension, int distance, int imageDimension, int pixelBinning,
                                           int binning) {
        if (distance >= 0) {
            int effectiveWidth = pixelDimension * pixelBinning + binning;
            double adjustedDistance = distance - (imageDimension - effectiveWidth);
            return (int) (pixelDimension - adjustedDistance / pixelBinning);
        }

        return pixelDimension;
    }

    /**
     * Calculates the maximum cursor position in the image.
     *
     * @return the calculated maximum cursor position
     */
    public Point getMaxCursorPosition(ImagePlus img) {
        Dimension usefulArea = getUsefulArea(img);
        Point pixelBinning = getPixelBinning();

        return new Point(
                calculateMaxCursorPosition(usefulArea.width, CCF.width, img.getWidth(), pixelBinning.x, binning.x),
                calculateMaxCursorPosition(usefulArea.height, CCF.height, img.getHeight(), pixelBinning.y, binning.y));
    }

    /**
     * Multiplication factor to determine positions in the image window; it is 1 for overlap, otherwise equal to
     * binning
     *
     * @return the binning to use in the image window
     */
    public Point getPixelBinning() {
        if (overlap) {
            return new Point(1, 1);
        } else {
            return binning;
        }
    }

    // Getters and setters for various parameters follow, allowing external modification and access to the settings.
    // These include straightforward implementations to set and get values for pixel size, magnification, NA, etc.
    // Some setters parse strings to double values, enabling easy handling of text input.
    // Additionally, there are methods to get and set the binning and CCF dimensions as strings for user interfaces.
    public double getPixelSize() {
        return pixelSize;
    }

    public void setPixelSize(String pixelSize) {
        this.pixelSize = Double.parseDouble(pixelSize) / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getPixelSizeInterface() {
        return pixelSize * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
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
        this.emLambda = Double.parseDouble(emLambda) / NANO_CONVERSION_FACTOR;
    }

    public double getEmLambdaInterface() {
        return emLambda * NANO_CONVERSION_FACTOR;
    }

    public double getSigma2() {
        return sigma2;
    }

    public void setSigma2(String sigma2) {
        this.sigma2 = Double.parseDouble(sigma2);
    }

    public double getEmLambda2() {
        return emLambda2;
    }

    public void setEmLambda2(String emLambda2) {
        this.emLambda2 = Double.parseDouble(emLambda2) / NANO_CONVERSION_FACTOR;
    }

    public double getEmLambda2Interface() {
        return emLambda2 * NANO_CONVERSION_FACTOR;
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

    public double getParamAx() {
        return paramAx;
    }

    public double getParamAxInterface() {
        return paramAx * NANO_CONVERSION_FACTOR;
    }

    public double getParamAy() {
        return paramAy;
    }

    public double getParamAyInterface() {
        return paramAy * NANO_CONVERSION_FACTOR;
    }

    public double getParamW() {
        return paramW;
    }

    public double getParamW2() {
        return paramW2;
    }

    public double getParamW2Interface() {
        return paramW2 * NANO_CONVERSION_FACTOR;
    }

    public double getParamWInterface() {
        return paramW * NANO_CONVERSION_FACTOR;
    }

    public double getParamZ() {
        return paramZ;
    }

    public double getParamZInterface() {
        return paramZ * NANO_CONVERSION_FACTOR;
    }

    public double getParamZ2() {
        return paramZ2;
    }

    public double getParamZ2Interface() {
        return paramZ2 * NANO_CONVERSION_FACTOR;
    }

    public double getParamRx() {
        return paramRx;
    }

    public double getParamRxInterface() {
        return paramRx * NANO_CONVERSION_FACTOR;
    }

    public double getParamRy() {
        return paramRy;
    }

    public double getParamRyInterface() {
        return paramRy * NANO_CONVERSION_FACTOR;
    }

    public Point getBinning() {
        return binning;
    }

    public void setBinning(String binning) {
        String[] parts = binning.replace(" ", "").split("x");
        int binningX = Integer.parseInt(parts[0]);
        int binningY = Integer.parseInt(parts[1]);

        if (binningX < 1 || binningY < 1) {
            throw new InvalidUserInputException("Binning can't be smaller than 1.");
        }

        this.binning.x = binningX;
        this.binning.y = binningY;
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

    public int getFirstFrame() {
        return firstFrame;
    }

    public void setFirstFrame(String firstFrame) {
        int intFirstFrame = Integer.parseInt(firstFrame);
        if (intFirstFrame >= lastFrame || intFirstFrame < 1) {
            throw new InvalidUserInputException(
                    "First frame set incorrectly, it needs to be between 1 and last frame" + ".");
        }

        this.firstFrame = intFirstFrame;
    }

    public int getLastFrame() {
        return lastFrame;
    }

    public void setLastFrame(String lastFrame) {
        int intLastFrame = Integer.parseInt(lastFrame);
        if (intLastFrame <= firstFrame || intLastFrame < 1) {
            throw new InvalidUserInputException("Last frame set incorrectly, it needs to be larger than first frame.");
        }

        this.lastFrame = intLastFrame;
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

    public boolean isFCCSDisp() {
        return FCCSDisp;
    }

    public void setFCCSDisp(String FCCSDisp) {
        this.FCCSDisp = Boolean.parseBoolean(FCCSDisp);
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

    public boolean isOverlap() {
        return overlap;
    }

    public void setOverlap(boolean overlap) {
        this.overlap = overlap;
    }

    public boolean isMSD3d() {
        return MSD3d;
    }

    public void setMSD3d(boolean isMSD3d) {
        this.MSD3d = isMSD3d;
    }

    public boolean isMSD() {
        return MSD;
    }

    public void setMSD(boolean MSD) {
        this.MSD = MSD;
    }

    public int getChannelNumber() {
        return channelNumber;
    }

    public int getLagGroupNumber() {
        return lagGroupNumber;
    }

    public int getSlidingWindowLength() {
        return slidingWindowLength;
    }

    public void setSlidingWindowLength(int slidingWindowLength) {
        this.slidingWindowLength = slidingWindowLength;
    }
}
