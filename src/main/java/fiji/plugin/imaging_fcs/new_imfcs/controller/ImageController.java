package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.OptionsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.MeanSquareDisplacement;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.SelectedPixel;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Range;
import fiji.plugin.imaging_fcs.new_imfcs.view.ImageView;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageCanvas;
import ij.gui.ImageWindow;
import ij.gui.Roi;

import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The ImageController class handles the interactions with the image data,
 * managing the loading, processing, and analysis of images for FCS data.
 */
public final class ImageController {
    private final ImageModel imageModel;
    private final MainPanelController mainPanelController;
    private final BackgroundSubtractionController backgroundSubtractionController;
    private final BleachCorrectionModel bleachCorrectionModel;
    private final Correlator correlator;
    private final FitController fitController;
    private final ExpSettingsModel settings;
    private final OptionsModel options;
    private ImageView imageView;
    private int previousX = -1;
    private int previousY = -1;

    /**
     * Constructs a new ImageController with the given dependencies.
     *
     * @param mainPanelController             The main panel controller.
     * @param imageModel                      The image model.
     * @param backgroundSubtractionController The background subtraction controller.
     * @param fitController                   The fit controller.
     * @param bleachCorrectionModel           The bleach correction model.
     * @param correlator                      The correlator.
     * @param settings                        The experimental settings model.
     * @param options                         The options model.
     */
    public ImageController(MainPanelController mainPanelController, ImageModel imageModel,
                           BackgroundSubtractionController backgroundSubtractionController,
                           FitController fitController, BleachCorrectionModel bleachCorrectionModel,
                           Correlator correlator, ExpSettingsModel settings, OptionsModel options) {
        this.mainPanelController = mainPanelController;
        this.imageModel = imageModel;
        this.backgroundSubtractionController = backgroundSubtractionController;
        this.fitController = fitController;
        this.bleachCorrectionModel = bleachCorrectionModel;
        this.correlator = correlator;
        this.settings = settings;
        this.options = options;
        imageView = null;
    }

    /**
     * Checks if an image is currently loaded in the model.
     *
     * @return true if an image is loaded, false otherwise.
     */
    public boolean isImageLoaded() {
        return imageModel.isImageLoaded();
    }

    /**
     * Retrieves the currently loaded image from the model.
     *
     * @return The ImagePlus object representing the currently loaded image.
     */
    public ImagePlus getImage() {
        return imageModel.getImage();
    }

    /**
     * Loads the given image into the model and sets up the view and event listeners.
     *
     * @param image The ImagePlus instance to load.
     */
    public void loadImage(ImagePlus image) {
        // reset the results if we load a new image
        correlator.resetResults();
        Plots.closePlots();

        imageModel.loadImage(image);

        imageView = new ImageView();
        imageView.showImage(imageModel);

        image.getCanvas().addMouseListener(imageMouseClicked());
        image.getCanvas().addKeyListener(imageKeyPressed());

        ImageModel.adaptImageScale(image);

        mainPanelController.setLastFrame(imageModel.getStackSize());
        backgroundSubtractionController.setTfBackground(imageModel.getBackground());
        backgroundSubtractionController.setTfBackground2(imageModel.getBackground2());
    }

    /**
     * Unloads the current image from the model.
     */
    public void unloadImage() {
        imageModel.unloadImage();
    }

    /**
     * Performs the correlation and fitting for a pixel at the given coordinates.
     *
     * @param x                      The x-coordinate of the pixel.
     * @param y                      The y-coordinate of the pixel.
     * @param singlePixelCorrelation A boolean indicating if it is a single pixel correlation.
     * @return An array of Points representing the cursor positions.
     */
    private Point[] correlatePixel(int x, int y, boolean singlePixelCorrelation) {
        SelectedPixel selectedPixel = new SelectedPixel(imageModel, correlator, settings);
        try {
            Point[] cursorPositions = selectedPixel.performCorrelationFunctionEvaluation(x, y, singlePixelCorrelation);
            Point pixel = cursorPositions[0];
            PixelModel pixelModel = correlator.getPixelModel(pixel.x, pixel.y);

            fitController.fit(pixelModel, correlator.getLagTimes(), correlator.getRegularizedCovarianceMatrix(), x, y);

            if (settings.isMSD()) {
                pixelModel.setMSD(MeanSquareDisplacement.correlationToMSD(pixelModel.getAcf(), settings.getParamAx(),
                        settings.getParamAy(), settings.getParamW(), settings.getSigmaZ(), settings.isMSD3d()));
            }

            return cursorPositions;
        } catch (RuntimeException e) {
            IJ.showMessage("Error", e.getMessage());
        }

        return null;
    }

    /**
     * Correlates a single pixel at the specified coordinates and plots the results.
     *
     * @param x The x-coordinate of the pixel.
     * @param y The y-coordinate of the pixel.
     */
    private void correlateSinglePixelAndPlot(int x, int y) {
        Point[] cursorPositions = correlatePixel(x, y, true);
        if (cursorPositions == null) {
            return;
        }

        try {
            plotResuts(cursorPositions);
            plotFittedParams(cursorPositions);
        } catch (RuntimeException e) {
            IJ.showMessage("Plot error", e.getMessage());
        }
    }

    /**
     * Plots multiple PixelModels.
     *
     * @param pixelModels A list of PixelModels to plot.
     */
    private void plotMultiplePixelsModels(List<PixelModel> pixelModels) {
        Plots.plotCorrelationFunction(pixelModels, correlator.getLagTimes(), null, settings.getBinning(),
                settings.getCCF(), fitController.getFitStart(), fitController.getFitEnd());
        if (settings.isMSD()) {
            Plots.plotMSD(pixelModels, correlator.getLagTimes(), null, settings.getBinning());
        }
    }

    /**
     * Correlates a Region of Interest (ROI) in the image.
     *
     * @param imgRoi The ROI to be correlated.
     */
    public void correlateROI(Roi imgRoi) {
        Point pixelBinning = settings.getPixelBinning();
        Rectangle rect = imgRoi.getBounds();

        Range xRange, yRange;

        try {
            xRange = new Range((int) Math.ceil(rect.x / (double) pixelBinning.x),
                    (int) Math.ceil((rect.x + rect.width - settings.getBinning().x) / (double) pixelBinning.x), 1);
            yRange = new Range((int) Math.ceil(rect.y / (double) pixelBinning.y),
                    (int) Math.ceil((rect.y + rect.height - settings.getBinning().y) / (double) pixelBinning.y), 1);
        } catch (IllegalArgumentException e) {
            // catch if the range isn't valid
            IJ.showMessage("ROI does not cover a whole single pixel in the binned image.");
            return;
        }

        List<PixelModel> correlatedPixels = new ArrayList<>();

        for (int x = xRange.getStart(); x <= xRange.getEnd(); x += xRange.getStep()) {
            for (int y = yRange.getStart(); y <= yRange.getEnd(); y += yRange.getStep()) {
                Point[] points = correlatePixel(x, y, false);
                if (points == null) {
                    IJ.log(String.format("Fail to correlate points for x=%d, y=%d", x, y));
                    continue;
                }

                PixelModel pixelModel = correlator.getPixelModel(points[0].x, points[0].y);
                correlatedPixels.add(pixelModel);

                plotFittedParams(points);
            }
            IJ.showProgress(x - xRange.getStart(), xRange.length());
        }

        plotMultiplePixelsModels(correlatedPixels);
    }

    /**
     * Validates if the given ROI is within the image bounds.
     *
     * @param roi The Region of Interest to check.
     * @return {@code true} if the ROI is within bounds, {@code false} otherwise.
     */
    public boolean isROIValid(Roi roi) {
        return imageModel.isROIValid(roi);
    }

    /**
     * Plots the results for a pixel at the given coordinates.
     *
     * @param cursorPositions The array of cursor positions where the two first elements represent the pixel
     *                        coordinates.
     */
    private void plotResuts(Point[] cursorPositions) {
        Point p = cursorPositions[0];
        PixelModel pixelModel = correlator.getPixelModel(p.x, p.y);

        if (options.isPlotACFCurves()) {
            Plots.plotCorrelationFunction(Collections.singletonList(pixelModel), correlator.getLagTimes(),
                    cursorPositions, settings.getBinning(), settings.getCCF(), fitController.getFitStart(),
                    fitController.getFitEnd());
        }

        if (options.isPlotSDCurves()) {
            Plots.plotStandardDeviation(pixelModel.getStandardDeviationAcf(), correlator.getLagTimes(), p);
        }

        if (options.isPlotIntensityCurves()) {
            Plots.plotIntensityTrace(bleachCorrectionModel.getIntensityTrace1(),
                    bleachCorrectionModel.getIntensityTrace2(), bleachCorrectionModel.getIntensityTime(),
                    cursorPositions);
        }

        if (options.isPlotBlockingCurve()) {
            Plots.plotBlockingCurve(correlator.getVarianceBlocks(), correlator.getBlockIndex());
        }

        if (options.isPlotCovMats() && fitController.isGLS()) {
            Plots.plotCovarianceMatrix(correlator.getRegularizedCovarianceMatrix());
        }

        if (settings.isMSD()) {
            Plots.plotMSD(Collections.singletonList(pixelModel), correlator.getLagTimes(), p, settings.getBinning());
        }

        if (options.isPlotResCurves() && pixelModel.isFitted()) {
            Plots.plotResiduals(pixelModel.getResiduals(), correlator.getLagTimes(), p);
        }
    }

    /**
     * Plots the fitted parameters at the specified cursor positions.
     * This method retrieves the pixel model at the given position, checks if it has been fitted,
     * and if so, plots the parameter maps. Additionally, if the option is enabled, it plots the
     * parameter histograms.
     *
     * @param cursorPositions An array of Points representing the cursor positions.
     */
    private void plotFittedParams(Point[] cursorPositions) {
        Point p = cursorPositions[0];
        PixelModel pixelModel = correlator.getPixelModel(p.x, p.y);

        if (pixelModel.isFitted()) {
            Point minimumPosition = settings.getMinCursorPosition();
            Point maximumPosition = settings.getMaxCursorPosition(imageModel.getImage());

            ImagePlus imgParams =
                    Plots.plotParameterMaps(pixelModel, p, minimumPosition, maximumPosition, settings.getPixelBinning(),
                            imageParamClicked());

            if (options.isPlotParaHist()) {
                Plots.plotParamHistogramWindow(imgParams);
            }
        }
    }

    /**
     * Creates a MouseListener to handle mouse clicks on the image canvas.
     *
     * @return The MouseListener instance.
     */
    public MouseListener imageMouseClicked() {
        return new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent event) {
                int x = imageModel.getCanvas().offScreenX(event.getX());
                int y = imageModel.getCanvas().offScreenY(event.getY());

                if (!settings.isOverlap()) {
                    x /= settings.getBinning().x;
                    y /= settings.getBinning().y;

                    previousX = x * settings.getBinning().x;
                    previousY = y * settings.getBinning().y;
                }

                correlateSinglePixelAndPlot(x, y);
            }
        };
    }

    /**
     * Creates a KeyListener to handle key presses on the image canvas.
     *
     * @return The KeyListener instance.
     */
    public KeyListener imageKeyPressed() {
        return new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent event) {
                Roi roi = imageModel.getRoi();
                if (roi != null) {
                    int x = (int) roi.getXBase();
                    int y = (int) roi.getYBase();

                    if (!settings.isOverlap()) {
                        // Move the roi by the size of the binning
                        x = previousX + (x - previousX) * settings.getBinning().x;
                        y = previousY + (y - previousY) * settings.getBinning().y;

                        x /= settings.getBinning().x;
                        y /= settings.getBinning().y;

                        // Here we multiply by the binning again to make sure that X and Y are factor of the binning
                        // It is only useful if the binning was changed by the user in the meantime
                        previousX = x * settings.getBinning().x;
                        previousY = y * settings.getBinning().y;
                    }

                    correlateSinglePixelAndPlot(x, y);
                }
            }
        };
    }

    /**
     * Creates and returns a MouseListener that responds to mouse clicks on an ImageCanvas.
     * This listener retrieves pixel coordinates from the clicked location, checks if the pixel
     * model is correlated, updates fit parameters on the view, and plots the results.
     *
     * @return a MouseListener that handles mouse click events on an ImageCanvas.
     */
    public MouseListener imageParamClicked() {
        return new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent event) {
                Object source = event.getSource();

                if (source instanceof ImageCanvas) {
                    ImageCanvas canvas = (ImageCanvas) source;
                    ImageWindow window = (ImageWindow) canvas.getParent();
                    ImagePlus img = window.getImagePlus();

                    // reset ROI
                    img.deleteRoi();

                    int x = canvas.offScreenX(event.getX());
                    int y = canvas.offScreenY(event.getY());

                    // The pixel is not correlated on this pixel
                    if (Double.isNaN(img.getStack().getProcessor(1).getPixelValue(x, y))) {
                        return;
                    }

                    Point pixelBinning = settings.getPixelBinning();
                    Point minimumPosition = settings.getMinCursorPosition();

                    x = (x + minimumPosition.x) * pixelBinning.x;
                    y = (y + minimumPosition.y) * pixelBinning.y;
                    PixelModel pixelModel = correlator.getPixelModel(x, y);

                    // if the pixel model is not correlated we do not plot
                    if (pixelModel == null || pixelModel.getAcf() == null) {
                        return;
                    }

                    if (pixelModel.isFitted()) {
                        fitController.updateFitParams(pixelModel.getFitParams());
                    }

                    int x2 = x + settings.getCCF().width;
                    int y2 = y + settings.getCCF().height;

                    Point[] points = new Point[]{new Point(x, y), new Point(x2, y2)};

                    plotResuts(points);
                }
            }

            @Override
            public void mouseReleased(MouseEvent event) {
                Object source = event.getSource();

                if (source instanceof ImageCanvas) {
                    ImageCanvas canvas = (ImageCanvas) source;
                    ImageWindow window = (ImageWindow) canvas.getParent();
                    ImagePlus img = window.getImagePlus();

                    Roi roi = img.getRoi();

                    if (roi != null && roi.getFeretsDiameter() > 1) {
                        roi.setStrokeColor(Color.BLUE);

                        Point pixelBinning = settings.getPixelBinning();
                        Point minimumPosition = settings.getMinCursorPosition();

                        List<PixelModel> pixelModels =
                                SelectedPixel.getPixelModelsInRoi(roi, pixelBinning, minimumPosition,
                                        correlator.getPixelsModel());

                        plotMultiplePixelsModels(pixelModels);
                    }
                }
            }
        };
    }
}
