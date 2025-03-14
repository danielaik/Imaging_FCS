package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.InvalidUserInputException;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Range;

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
    // Callback to use when a setting is changed
    private final Runnable resetCallback;
    private double pixelSize = 24 / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    private double magnification = 100;
    private double NA = 1.49;
    private double sigma = 0.8;
    private double emLambda = 515 / NANO_CONVERSION_FACTOR;
    private double sigma2 = 0.8;
    private double emLambda2 = 600 / NANO_CONVERSION_FACTOR;
    private double sigmaZ = 1;
    private double sigmaZ2 = 1;
    //// Other user parameters
    private int firstFrame = 1;
    private int lastFrame = 0;
    private double frameTime = 0.001;
    private int correlatorP = 16;
    private int correlatorQ = 8;
    private String fitModel = Constants.ITIR_FCS_2D;
    private String paraCor = "N vs D";
    private String dCCF = Constants.X_DIRECTION;
    private String bleachCorrection = Constants.NO_BLEACH_CORRECTION;
    private String filter = Constants.NO_FILTER;
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
     * Constructs an ExpSettingsModel with default settings and a no-op reset callback.
     * Initializes settings by calling {@code updateSettings()} and {@code updateChannelNumber()}.
     */
    public ExpSettingsModel() {
        updateSettings();
        updateChannelNumber();

        this.resetCallback = () -> {};
    }

    /**
     * Constructs an ExpSettingsModel with a specified reset callback.
     * Initializes settings by calling {@code updateSettings()} and {@code updateChannelNumber()}.
     *
     * @param resetCallback a {@link Runnable} executed when a reset is needed.
     */
    public ExpSettingsModel(Runnable resetCallback) {
        updateSettings();
        updateChannelNumber();

        this.resetCallback = resetCallback;
    }

    /**
     * Copy constructor for ExpSettingsModel.
     * Creates a new instance by copying the values from the provided instance.
     * Do not copy the binning and the CCF to restart them
     *
     * @param other the ExpSettingsModel instance to copy from.
     */
    public ExpSettingsModel(ExpSettingsModel other) {
        this.pixelSize = other.pixelSize;
        this.magnification = other.magnification;
        this.NA = other.NA;
        this.sigma = other.sigma;
        this.emLambda = other.emLambda;
        this.sigma2 = other.sigma2;
        this.emLambda2 = other.emLambda2;
        this.sigmaZ = other.sigmaZ;
        this.sigmaZ2 = other.sigmaZ2;
        this.firstFrame = other.firstFrame;
        this.lastFrame = other.lastFrame;
        this.frameTime = other.frameTime;
        this.correlatorP = other.correlatorP;
        this.correlatorQ = other.correlatorQ;
        this.fitModel = other.fitModel;
        this.paraCor = other.paraCor;
        this.dCCF = other.dCCF;
        this.bleachCorrection = other.bleachCorrection;
        this.filter = other.filter;
        this.filterLowerLimit = other.filterLowerLimit;
        this.filterUpperLimit = other.filterUpperLimit;
        this.FCCSDisp = other.FCCSDisp;
        this.overlap = other.overlap;
        this.MSD3d = other.MSD3d;
        this.MSD = other.MSD;
        this.slidingWindowLength = other.slidingWindowLength;
        this.channelNumber = other.channelNumber;
        this.lagGroupNumber = other.lagGroupNumber;

        setBinning(other.binning);
        setCCF(other.CCF);

        // Reset callback should not be copied directly.
        // It is assumed the new instance has a no-op reset callback, or you could pass it as a parameter.
        this.resetCallback = () -> {};

        // Update settings after copying to ensure consistency.
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
        data.put("Correlator P", getCorrelatorP());
        data.put("Correlator Q", getCorrelatorQ());
        data.put("Sigma", getSigma());
        data.put("Sigma 2", getSigma2());
        data.put("Sigma Z", getSigmaZ());
        data.put("Sigma Z 2", getSigmaZ2());
        data.put("Lambda", getEmLambdaInterface());
        data.put("Lambda 2", getEmLambda2Interface());
        data.put("Frame time", getFrameTime());

        data.put("Overlap", isOverlap());

        return data;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> data = toMapConfig();

        data.put("First frame", getFirstFrame());
        data.put("Last frame", getLastFrame());
        data.put("Fit model", getFitModel());
        data.put("FCCS Display", isFCCSDisp());
        data.put("Bleach correction", getBleachCorrection());
        data.put("Sliding window length", getSlidingWindowLength());
        data.put("Filter", getFilter());
        data.put("Filter Lower Limit", getFilterLowerLimit());
        data.put("Filter Upper Limit", getFilterUpperLimit());
        data.put("MSD", isMSD());
        data.put("MSD 3D", isMSD3d());

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
        setCorrelatorP(data.get("Correlator P").toString());
        setCorrelatorQ(data.get("Correlator Q").toString());
        setSigma(data.get("Sigma").toString());
        setSigma2(data.get("Sigma 2").toString());
        setSigmaZ(data.get("Sigma Z").toString());
        setSigmaZ2(data.get("Sigma Z 2").toString());
        setEmLambda(data.get("Lambda").toString());
        setEmLambda2(data.get("Lambda 2").toString());
        setFrameTime(data.get("Frame time").toString());

        setOverlap(Boolean.parseBoolean(data.get("Overlap").toString()));

        // Update the settings accordingly.
        updateSettings();
        updateChannelNumber();
    }

    /**
     * Loads additional settings from a map, specifically for Excel loading.
     * Extends the `fromMap` method to include settings related to frame range, fit model,
     * bleach correction, filtering, and sliding window parameters.
     *
     * @param data a map containing the settings to be loaded, with keys corresponding to parameter names.
     */
    public void fromMapExcelLoading(Map<String, Object> data) {
        fromMap(data);

        setLastFrame(data.get("Last frame").toString());
        setFirstFrame(data.get("First frame").toString());
        setFitModel(data.get("Fit model").toString());
        setBleachCorrection(data.get("Bleach correction").toString());
        setFilter(data.get("Filter").toString());

        setFCCSDisp(Boolean.parseBoolean(data.get("FCCS Display").toString()));
        setMSD(Boolean.parseBoolean(data.get("MSD").toString()));
        setMSD3d(Boolean.parseBoolean(data.get("MSD 3D").toString()));

        setSlidingWindowLength(Integer.parseInt(data.get("Sliding window length").toString()));
        setFilterLowerLimit(Integer.parseInt(data.get("Filter Lower Limit").toString()));
        setFilterUpperLimit(Integer.parseInt(data.get("Filter Upper Limit").toString()));
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
     * Determines the useful area of the image that can be used for correlation, depending on whether overlap is
     * allowed.
     * This method calculates the dimensions of the area based on the binning factors and the overlap setting.
     *
     * @param dimension the dimension of the image (width and height)
     * @return the dimension of the useful area as a Dimension object
     */
    public Dimension getUsefulArea(Dimension dimension) {
        if (overlap) {
            return new Dimension(dimension.width - Math.abs(CCF.width) + 1,
                    dimension.height - Math.abs(CCF.height) + 1);
        } else {
            return new Dimension((dimension.width / binning.x) - 1, (dimension.height / binning.y) - 1);
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
            return (int) (pixelDimension - adjustedDistance / binning);
        }

        return pixelDimension;
    }

    /**
     * Calculates the maximum cursor position in the image based on the given dimensions and pixel binning factors.
     * The method considers the useful area of the image and applies the pixel binning to determine the maximum
     * cursor positions along both the x and y axes.
     *
     * @param dimension the dimension of the image (width and height)
     * @return the calculated maximum cursor position as a Point object
     */
    public Point getMaxCursorPosition(Dimension dimension) {
        Dimension usefulArea = getUsefulArea(dimension);
        Point pixelBinning = getPixelBinning();

        return new Point(
                calculateMaxCursorPosition(usefulArea.width, CCF.width, dimension.width, pixelBinning.x, binning.x),
                calculateMaxCursorPosition(usefulArea.height, CCF.height, dimension.height, pixelBinning.y, binning.y));
    }

    public Range[] getAllArea(Dimension dimension) {
        Point startLocation = getMinCursorPosition();
        Point endLocation = getMaxCursorPosition(dimension);
        Point pixelBinning = getPixelBinning();

        int startX = startLocation.x * pixelBinning.x;
        int width = (endLocation.x - startLocation.x + 1) * pixelBinning.x;
        int startY = startLocation.y * pixelBinning.y;
        int height = (endLocation.y - startLocation.y + 1) * pixelBinning.y;

        return new Range[]{
                new Range(startX, width, pixelBinning.x), new Range(startY, height, pixelBinning.y)
        };
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

    /**
     * Converts a point to its corresponding binning scale, adjusted by the minimum position offset.
     *
     * @param p The point to be converted.
     * @return A new Point object representing the converted coordinates.
     */
    public Point convertPointToBinning(Point p) {
        Point pixelBinning = getPixelBinning();
        Point minimumPosition = getMinCursorPosition();

        return new Point(p.x / pixelBinning.x - minimumPosition.x, p.y / pixelBinning.y - minimumPosition.y);
    }

    /**
     * Calculates the dimensions of an image based on the minimum and maximum positions within the image.
     *
     * @param imageDimension The original dimensions of the image.
     * @return A Dimension object representing the adjusted image dimensions.
     */
    public Dimension getConvertedImageDimension(Dimension imageDimension) {
        Point minimumPosition = getMinCursorPosition();
        Point maximumPosition = getMaxCursorPosition(imageDimension);

        int width = (maximumPosition.x - minimumPosition.x + 1);
        int height = (maximumPosition.y - minimumPosition.y + 1);

        if (overlap) {
            width = width - binning.x + 1;
            height = height - binning.y + 1;
        }

        return new Dimension(width, height);
    }

    // Getters and setters for various parameters follow, allowing external modification and access to the settings.
    // These include straightforward implementations to set and get values for pixel size, magnification, NA, etc.
    // Some setters parse strings to double values, enabling easy handling of text input.
    // Additionally, there are methods to get and set the binning and CCF dimensions as strings for user interfaces.
    public double getPixelSize() {
        return pixelSize;
    }

    public void setPixelSize(String pixelSize) {
        double tmp = Double.parseDouble(pixelSize);
        resetCallback.run();
        this.pixelSize = tmp / PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getPixelSizeInterface() {
        return pixelSize * PIXEL_SIZE_REAL_SPACE_CONVERSION_FACTOR;
    }

    public double getMagnification() {
        return magnification;
    }

    public void setMagnification(String magnification) {
        double tmp = Double.parseDouble(magnification);
        resetCallback.run();
        this.magnification = tmp;
    }

    public double getNA() {
        return NA;
    }

    public void setNA(String NA) {
        double tmp = Double.parseDouble(NA);
        resetCallback.run();
        this.NA = tmp;
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(String sigma) {
        double tmp = Double.parseDouble(sigma);
        resetCallback.run();
        this.sigma = tmp;
    }

    public double getEmLambda() {
        return emLambda;
    }

    public void setEmLambda(String emLambda) {
        double tmp = Double.parseDouble(emLambda);
        resetCallback.run();
        this.emLambda = tmp / NANO_CONVERSION_FACTOR;
    }

    public double getEmLambdaInterface() {
        return emLambda * NANO_CONVERSION_FACTOR;
    }

    public double getSigma2() {
        return sigma2;
    }

    public void setSigma2(String sigma2) {
        double tmp = Double.parseDouble(sigma2);
        resetCallback.run();
        this.sigma2 = tmp;
    }

    public double getEmLambda2() {
        return emLambda2;
    }

    public void setEmLambda2(String emLambda2) {
        double tmp = Double.parseDouble(emLambda2);
        resetCallback.run();
        this.emLambda2 = tmp / NANO_CONVERSION_FACTOR;
    }

    public double getEmLambda2Interface() {
        return emLambda2 * NANO_CONVERSION_FACTOR;
    }

    public double getSigmaZ() {
        return sigmaZ;
    }

    public void setSigmaZ(String sigmaZ) {
        double tmp = Double.parseDouble(sigmaZ);
        resetCallback.run();
        this.sigmaZ = tmp;
    }

    public double getSigmaZ2() {
        return sigmaZ2;
    }

    public void setSigmaZ2(String sigmaZ2) {
        double tmp = Double.parseDouble(sigmaZ2);
        resetCallback.run();
        this.sigmaZ2 = tmp;
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

        resetCallback.run();

        this.binning.x = binningX;
        this.binning.y = binningY;
    }

    public void setBinning(Point binning) {
        this.binning.x = binning.x;
        this.binning.y = binning.y;
    }

    public String getBinningString() {
        return String.format("%d x %d", binning.x, binning.y);
    }

    public Dimension getCCF() {
        return CCF;
    }

    public void setCCF(String CCF) {
        String[] parts = CCF.replace(" ", "").split("x");
        int width = Integer.parseInt(parts[0]);
        int height = Integer.parseInt(parts[1]);

        resetCallback.run();

        this.CCF.width = width;
        this.CCF.height = height;
    }

    public void setCCF(Dimension CCF) {
        this.CCF.width = CCF.width;
        this.CCF.height = CCF.height;
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
                    "First frame set incorrectly, it needs to be between 1 and last frame.");
        }

        resetCallback.run();

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

        resetCallback.run();

        this.lastFrame = intLastFrame;
    }

    public double getFrameTime() {
        return frameTime;
    }

    public void setFrameTime(String frameTime) {
        double tmp = Double.parseDouble(frameTime);
        resetCallback.run();
        this.frameTime = tmp;
    }

    public int getCorrelatorP() {
        return correlatorP;
    }

    public void setCorrelatorP(String correlatorP) {
        int tmp = Integer.parseInt(correlatorP);
        resetCallback.run();
        this.correlatorP = tmp;
    }

    public int getCorrelatorQ() {
        return correlatorQ;
    }

    public void setCorrelatorQ(String correlatorQ) {
        int tmp = Integer.parseInt(correlatorQ);
        resetCallback.run();
        this.correlatorQ = tmp;
    }

    public String getFitModel() {
        return fitModel;
    }

    public void setFitModel(String fitModel) {
        resetCallback.run();
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
        resetCallback.run();
        this.bleachCorrection = bleachCorrection;
    }

    public String getFilter() {
        return filter;
    }

    public void setFilter(String filter) {
        resetCallback.run();
        this.filter = filter;
    }

    public boolean isFCCSDisp() {
        return FCCSDisp;
    }

    public void setFCCSDisp(boolean FCCSDisp) {
        this.FCCSDisp = FCCSDisp;
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
