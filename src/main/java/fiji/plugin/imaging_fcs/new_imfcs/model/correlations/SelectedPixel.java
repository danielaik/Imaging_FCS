package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import fiji.plugin.imaging_fcs.new_imfcs.controller.FitController;
import fiji.plugin.imaging_fcs.new_imfcs.enums.FitFunctions;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import ij.gui.Overlay;
import ij.gui.Roi;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * The SelectedPixel class handles the operations related to pixel selection and correlation for image processing.
 */
public class SelectedPixel {
    private final ImageModel imageModel;
    private final Correlator correlator;
    private final ExpSettingsModel settings;

    /**
     * Constructs a SelectedPixel instance with the given models and settings.
     *
     * @param imageModel the image model
     * @param correlator the correlator
     * @param settings   the experimental settings model
     */
    public SelectedPixel(ImageModel imageModel, Correlator correlator, ExpSettingsModel settings) {
        this.imageModel = imageModel;
        this.correlator = correlator;
        this.settings = settings;
    }

    /**
     * Retrieves a list of PixelModel objects within the specified Region of Interest (ROI).
     * If the ROI is null, it retrieves all valid PixelModel objects from the provided pixelModels array.
     *
     * @param roi                   The Region of Interest within which to retrieve PixelModel objects. If null, all
     *                              PixelModel objects are considered.
     * @param pixelBinning          The binning factor applied to the pixel coordinates.
     * @param minimumPosition       The minimum position offset to apply to the pixel coordinates.
     * @param convertPointToBinning Function to convert pixel coordinates to their corresponding binning points.
     * @param pixelModels           The 2D array of PixelModel objects from which to retrieve the models.
     * @param fitController         the controller responsible for filtering pixels based on fit criteria.
     * @return A list of PixelModel objects within the specified ROI.
     */
    public static List<PixelModel> getPixelModelsInRoi(Roi roi, Point pixelBinning, Point minimumPosition,
                                                       Function<Point, Point> convertPointToBinning,
                                                       PixelModel[][] pixelModels, FitController fitController) {
        List<PixelModel> pixelModelList = new ArrayList<>();

        if (roi == null) {
            for (int x = 0; x < pixelModels.length; x++) {
                for (int y = 0; y < pixelModels[0].length; y++) {
                    PixelModel currentPixelModel = pixelModels[x][y];
                    if (currentPixelModel != null && currentPixelModel.getCorrelationFunction() != null) {
                        Point binningPoint = convertPointToBinning.apply(new Point(x, y));
                        if (!fitController.needToFilter(currentPixelModel, binningPoint.x, binningPoint.y)) {
                            pixelModelList.add(currentPixelModel);
                        }
                    }
                }
            }
        } else {
            Rectangle rect = roi.getBounds();

            for (int x = rect.x; x < rect.x + rect.width; x++) {
                for (int y = rect.y; y < rect.y + rect.height; y++) {
                    Point convertedCoordinates = new Point((x + minimumPosition.x) * pixelBinning.x,
                            (y + minimumPosition.y) * pixelBinning.y);

                    PixelModel currentPixelModel = pixelModels[convertedCoordinates.x][convertedCoordinates.y];

                    if (currentPixelModel != null && currentPixelModel.getCorrelationFunction() != null) {
                        Point binningPoint = convertPointToBinning.apply(convertedCoordinates);
                        if (!fitController.needToFilter(currentPixelModel, binningPoint.x, binningPoint.y)) {
                            pixelModelList.add(currentPixelModel);
                        }
                    }
                }
            }
        }

        return pixelModelList;
    }


    /**
     * Evaluates the correlation function for the selected pixel at the given coordinates.
     * Converts the pixel positions to image space and checks if the pixel is within bounds,
     * if cross-correlation areas overlap, or if binning exceeds the image size.
     * Throws exceptions if any conditions are violated.
     *
     * @param x                      The x-coordinate of the pixel.
     * @param y                      The y-coordinate of the pixel.
     * @param singlePixelCorrelation Indicates if the evaluation is for a single pixel.
     * @return The positions of the pixels after conversion to image space.
     * @throws IllegalArgumentException If the pixel is out of bounds, cross-correlation areas overlap, or binning is
     *                                  too large.
     */
    public Point[] performCorrelationFunctionEvaluation(int x, int y, boolean singlePixelCorrelation) {
        Point[] cursorPositions = convertToImageSpace(x, y);
        Point cursorPosition1 = cursorPositions[0];
        Point cursorPosition2 = cursorPositions[1];

        if (!isPixelWithinImage(x, y)) {
            throw new IllegalArgumentException(String.format("x=%d, y=%d are out of the image.", x, y));
        }

        if (isOverlapInDCFCCS()) {
            throw new IllegalArgumentException("Cross-correlation areas overlap.");
        }

        if (isBinningLargerThanImageSize()) {
            throw new IllegalArgumentException("Parameter binning is larger than image size.");
        }

        processPixels(cursorPosition1, cursorPosition2, singlePixelCorrelation);
        return cursorPositions;
    }

    /**
     * Converts the given pixel coordinates to image space coordinates.
     *
     * @param x the x-coordinate of the pixel
     * @param y the y-coordinate of the pixel
     * @return an array containing the converted coordinates
     */
    private Point[] convertToImageSpace(int x, int y) {
        Point pixelBinning = settings.getPixelBinning();

        Point cursorPosition1 = new Point(x * pixelBinning.x, y * pixelBinning.y);
        Point cursorPosition2 =
                new Point(x * pixelBinning.x + settings.getCCF().width, y * pixelBinning.y + settings.getCCF().height);

        return new Point[]{cursorPosition1, cursorPosition2};
    }

    /**
     * Checks if the given pixel coordinates are within the image boundaries.
     *
     * @param x the x-coordinate of the pixel
     * @param y the y-coordinate of the pixel
     * @return true if the pixel is within the image, false otherwise
     */
    private boolean isPixelWithinImage(int x, int y) {
        Point minCursorPosition = settings.getMinCursorPosition();
        Point maxCursorPosition = settings.getMaxCursorPosition(imageModel.getDimension());

        return x >= minCursorPosition.x && x <= maxCursorPosition.x && y >= minCursorPosition.y &&
                y <= maxCursorPosition.y;
    }

    /**
     * Checks if there is an overlap in DC-FCCS (Dual-Color Fluorescence Cross-Correlation Spectroscopy) areas.
     *
     * @return true if there is an overlap, false otherwise
     */
    private boolean isOverlapInDCFCCS() {
        return settings.getFitModel() == FitFunctions.DC_FCCS_2D &&
                Math.abs(settings.getCCF().width) < settings.getBinning().x &&
                Math.abs(settings.getCCF().height) < settings.getBinning().y;
    }

    /**
     * Checks if the binning size exceeds the image dimensions.
     *
     * @return True if the binning size is larger than the image width or height.
     */
    private boolean isBinningLargerThanImageSize() {
        return settings.getBinning().x > imageModel.getWidth() || settings.getBinning().y > imageModel.getHeight();
    }

    /**
     * Processes the pixels at the given cursor positions by setting up ROIs if required
     * and performing the appropriate correlation based on the selected fit model.
     *
     * @param cursorPosition1        The first cursor position.
     * @param cursorPosition2        The second cursor position.
     * @param singlePixelCorrelation Indicates if the operation is for a single pixel.
     */
    private void processPixels(Point cursorPosition1, Point cursorPosition2, boolean singlePixelCorrelation) {
        if (singlePixelCorrelation) {
            setupROIs(cursorPosition1, cursorPosition2);
        }

        correlator.correlate(imageModel.getImage(), cursorPosition1.x, cursorPosition1.y, cursorPosition2.x,
                cursorPosition2.y, settings.getFirstFrame(), settings.getLastFrame());
    }

    /**
     * Sets up the Regions of Interest (ROIs) at the given cursor positions.
     *
     * @param cursorPosition1 the first cursor position
     * @param cursorPosition2 the second cursor position
     */
    private void setupROIs(Point cursorPosition1, Point cursorPosition2) {
        if (imageModel.getOverlay() != null) {
            imageModel.getOverlay().clear();
        }

        Roi roi1 = new Roi(cursorPosition1.x, cursorPosition1.y, settings.getBinning().x, settings.getBinning().y);
        roi1.setStrokeColor(Color.BLUE);
        imageModel.setRoi(roi1);

        if (settings.getCCF().width != 0 || settings.getCCF().height != 0 ||
                settings.getFitModel() == FitFunctions.DC_FCCS_2D) {
            Roi roi2 = new Roi(cursorPosition2.x, cursorPosition2.y, settings.getBinning().x, settings.getBinning().y);
            roi2.setStrokeColor(Color.RED);
            imageModel.setOverlay(new Overlay(roi2));
        }
    }
}
