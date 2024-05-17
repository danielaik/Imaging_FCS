package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import ij.gui.Overlay;
import ij.gui.Roi;

import java.awt.*;

public class SelectedPixel {
    private final ImageModel imageModel;
    private final BleachCorrectionModel bleachCorrectionModel;
    private final ExpSettingsModel settings;

    // determines the number of pixels that can be correlated, depending on whether overlap is allowed
    private final Dimension pixelsDimension = new Dimension(0, 0);
    private final Point minCursorPosition = new Point(1, 1);
    private final Point maxCursorPosition = new Point(1, 1);
    // Multiplication factor to determine positions in the image window; it is 1 for overlap, otherwise equal to binning
    private Point pixelBinning = new Point(1, 1);

    public SelectedPixel(ImageModel imageModel, BleachCorrectionModel bleachCorrectionModel,
                         ExpSettingsModel settings) {
        this.imageModel = imageModel;
        this.bleachCorrectionModel = bleachCorrectionModel;
        this.settings = settings;
    }

    public void performCFE(int x, int y, Plots plots) {
        Point[] cursorPositions = convertToImageSpace(x, y);
        Point cursorPosition1 = cursorPositions[0];
        Point cursorPosition2 = cursorPositions[1];

        if (!isPixelWithinImage(x, y)) {
            throw new RuntimeException("Pixel coordinates are out of the image");
        }

        if (isOverlapInDCFCCS()) {
            throw new RuntimeException("Cross-correlation areas overlap.");
        }

        processPixels(cursorPosition1, cursorPosition2, plots);
    }

    private Point[] convertToImageSpace(int x, int y) {
        if (settings.isOverlap()) {
            pixelsDimension.width = imageModel.getWidth() - settings.getBinning().x;
            pixelsDimension.height = imageModel.getHeight() - settings.getBinning().y;
        } else {
            pixelsDimension.width = imageModel.getWidth() / settings.getBinning().x - 1;
            pixelsDimension.height = imageModel.getHeight() / settings.getBinning().y - 1;
            pixelBinning = settings.getBinning();
        }

        minCursorPosition.x = calculateMinCursorPosition(settings.getCCF().width, pixelBinning.x);
        minCursorPosition.y = calculateMinCursorPosition(settings.getCCF().height, pixelBinning.y);

        maxCursorPosition.x = calculateMaxCursorPosition(pixelsDimension.width, settings.getCCF().width,
                imageModel.getWidth(), pixelBinning.x, settings.getBinning().x);
        maxCursorPosition.y = calculateMaxCursorPosition(pixelsDimension.height, settings.getCCF().height,
                imageModel.getHeight(), pixelBinning.y, settings.getBinning().y);

        Point cursorPosition1 = new Point(x * pixelBinning.x, y * pixelBinning.y);
        Point cursorPosition2 = new Point(x * pixelBinning.x + settings.getCCF().width,
                y * pixelBinning.y + settings.getCCF().height);

        return new Point[]{cursorPosition1, cursorPosition2};
    }

    private int calculateMaxCursorPosition(int pixelDimension, int distance, int imageDimension, int pixelBinning,
                                           int binning) {
        if (distance >= 0) {
            int effectiveWidth = pixelDimension * pixelBinning + binning;
            int adjustedDistance = distance - (imageDimension - effectiveWidth);
            return pixelDimension - adjustedDistance / pixelBinning;
        }

        return pixelDimension;
    }

    private int calculateMinCursorPosition(int distance, int pixelBinning) {
        if (distance < 0) {
            return -distance / pixelBinning;
        }
        return 0;
    }

    private boolean isPixelWithinImage(int x, int y) {
        return x >= minCursorPosition.x && x <= maxCursorPosition.x &&
                y >= minCursorPosition.y && y <= maxCursorPosition.y;
    }

    private boolean isOverlapInDCFCCS() {
        return settings.getFitModel().equals(Constants.DC_FCCS_2D) &&
                Math.abs(settings.getCCF().width) < settings.getBinning().x &&
                Math.abs(settings.getCCF().height) < settings.getBinning().y;
    }

    private void processPixels(Point cursorPosition1, Point cursorPosition2, Plots plots) {
        setupROIs(cursorPosition1, cursorPosition2);

        if (settings.getFitModel().equals(Constants.ITIR_FCS_2D) ||
                settings.getFitModel().equals(Constants.SPIM_FCS_3D)) {
            bleachCorrectionModel.calcIntensityTrace(imageModel.getImage(), cursorPosition1.x, cursorPosition1.y,
                    cursorPosition2.x, cursorPosition2.y, settings.getFirstFrame(), settings.getLastFrame());
            Correlator correlator = new Correlator(settings, bleachCorrectionModel, plots);

            correlator.correlate(imageModel.getImage(), cursorPosition1.x / pixelBinning.x,
                    cursorPosition1.y / pixelBinning.y, settings.getFirstFrame(), settings.getLastFrame());
        }
    }

    private void setupROIs(Point cursorPosition1, Point cursorPosition2) {
        if (imageModel.getOverlay() != null) {
            imageModel.getOverlay().clear();
        }

        Roi roi1 = new Roi(cursorPosition1.x, cursorPosition1.y, settings.getBinning().x, settings.getBinning().y);
        roi1.setStrokeColor(Color.BLUE);
        imageModel.setRoi(roi1);

        if (settings.getFitModel().equals(Constants.DC_FCCS_2D)) {
            Roi roi2 = new Roi(cursorPosition2.x, cursorPosition2.y, settings.getBinning().x, settings.getBinning().y);
            roi2.setStrokeColor(Color.RED);
            imageModel.setOverlay(new Overlay(roi2));
        }
    }
}
