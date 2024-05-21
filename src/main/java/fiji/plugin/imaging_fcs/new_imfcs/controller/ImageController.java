package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.SelectedPixel;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.ImageView;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

public final class ImageController {
    private final ImageModel imageModel;
    private final MainPanelController mainPanelController;
    private final BackgroundSubtractionController backgroundSubtractionController;
    private final BleachCorrectionModel bleachCorrectionModel;
    private final Correlator correlator;
    private final ExpSettingsModel settings;
    private ImageView imageView;
    private int previousX = -1;
    private int previousY = -1;

    public ImageController(MainPanelController mainPanelController, ImageModel imageModel,
                           BackgroundSubtractionController backgroundSubtractionController,
                           BleachCorrectionModel bleachCorrectionModel, Correlator correlator,
                           ExpSettingsModel settings) {
        this.mainPanelController = mainPanelController;
        this.imageModel = imageModel;
        this.backgroundSubtractionController = backgroundSubtractionController;
        this.bleachCorrectionModel = bleachCorrectionModel;
        this.correlator = correlator;
        this.settings = settings;
        imageView = null;
    }

    public boolean isImageLoaded() {
        return imageModel.isImageLoaded();
    }

    public void loadImage(ImagePlus image) {
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

    public MouseListener imageMouseClicked() {
        return new MouseListener() {
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

                SelectedPixel selectedPixel = new SelectedPixel(imageModel, bleachCorrectionModel, correlator, settings);
                try {
                    selectedPixel.performCFE(x, y);
                } catch (RuntimeException e) {
                    IJ.showMessage("Error", e.getMessage());
                }
            }

            @Override
            public void mousePressed(MouseEvent event) {

            }

            @Override
            public void mouseReleased(MouseEvent event) {

            }

            @Override
            public void mouseEntered(MouseEvent event) {

            }

            @Override
            public void mouseExited(MouseEvent event) {

            }
        };
    }

    public KeyListener imageKeyPressed() {
        return new KeyListener() {
            @Override
            public void keyTyped(KeyEvent event) {
            }

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

                    SelectedPixel selectedPixel =
                            new SelectedPixel(imageModel, bleachCorrectionModel, correlator, settings);
                    try {
                        selectedPixel.performCFE(x, y);
                    } catch (RuntimeException e) {
                        IJ.showMessage("Error", e.getMessage());
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent event) {
            }
        };
    }
}
