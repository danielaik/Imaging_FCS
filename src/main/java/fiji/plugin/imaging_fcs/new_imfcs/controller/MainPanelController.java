package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.*;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.*;
import fiji.plugin.imaging_fcs.new_imfcs.model.fit.BleachCorrectionModel;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Pair;
import fiji.plugin.imaging_fcs.new_imfcs.view.BleachCorrectionView;
import fiji.plugin.imaging_fcs.new_imfcs.view.ExpSettingsView;
import fiji.plugin.imaging_fcs.new_imfcs.view.MainPanelView;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.FilterLimitsSelectionView;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.MSDView;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.PolynomialOrderSelectionView;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.SlidingWindowSelectionView;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Overlay;
import ij.gui.Roi;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.ControllerUtils.getComboBoxSelectionFromEvent;
import static fiji.plugin.imaging_fcs.new_imfcs.controller.FocusListenerFactory.createFocusListener;

/**
 * Controller class for the main panel of the FCS plugin. It handles user interactions and coordinates
 * the update of models based on user input, as well as the update of views to reflect the current state
 * of the models.
 */
public final class MainPanelController {
    // Field declarations for the views and models that this controller will manage
    private final MainPanelView view;
    private final ExpSettingsView expSettingsView;
    private final BleachCorrectionView bleachCorrectionView;
    private final HardwareModel hardwareModel;
    private final OptionsModel optionsModel;
    private final ImageController imageController;
    private final ExpSettingsModel settings;
    private final BleachCorrectionModel bleachCorrectionModel;
    private final SimulationController simulationController;
    private final BackgroundSubtractionController backgroundSubtractionController;
    private final NBController nbController;
    private final FitController fitController;
    private final Correlator correlator;

    /**
     * Constructor that initializes models, views, and other controllers needed for the main panel.
     *
     * @param hardwareModel The model containing hardware settings for the imaging FCS analysis.
     */
    public MainPanelController(HardwareModel hardwareModel) {
        this.hardwareModel = hardwareModel;
        this.optionsModel = new OptionsModel(hardwareModel.isCuda());

        this.settings = new ExpSettingsModel();
        loadConfig();

        this.expSettingsView = new ExpSettingsView(this, settings);
        updateSettingsField();

        FitModel fitModel = new FitModel(settings);
        this.fitController = new FitController(fitModel);

        ImageModel imageModel = new ImageModel();
        this.backgroundSubtractionController = new BackgroundSubtractionController(imageModel);
        this.bleachCorrectionModel = new BleachCorrectionModel(settings, imageModel);
        this.correlator = new Correlator(settings, bleachCorrectionModel, fitModel);
        this.imageController = new ImageController(this, imageModel, backgroundSubtractionController, fitController,
                bleachCorrectionModel, correlator, settings, optionsModel);

        this.bleachCorrectionView = new BleachCorrectionView(this, bleachCorrectionModel);

        this.simulationController = new SimulationController(imageController, settings);

        this.nbController = new NBController(imageModel, settings, optionsModel, bleachCorrectionModel);


        this.view = new MainPanelView(this, this.settings);
    }

    /**
     * Constructs the file path to the configuration file in the user's home directory.
     *
     * @return the file path to the configuration file.
     */
    private String getConfigPath() {
        String userHomeDir = System.getProperty("user.home");
        String configFileName = ".ImFCS_config.yaml";
        return userHomeDir + FileSystems.getDefault().getSeparator() + configFileName;
    }

    /**
     * Loads the configuration from the YAML file located at the path specified by getConfigPath.
     * The method reads the file, parses the YAML content, and updates the settings and options models.
     * If an error occurs during reading or parsing, a message is logged.
     */
    private void loadConfig() {
        Yaml yaml = new Yaml();
        try (FileInputStream inputStream = new FileInputStream(getConfigPath())) {
            Map<String, Map<String, Object>> data = yaml.load(inputStream);
            settings.fromMap(data.get("Settings"));
            optionsModel.fromMap(data.get("Options"));
        } catch (Exception e) {
            IJ.log("Can't read configuration file.");
        }
    }

    /**
     * Sets the last frame for the analysis and updates related fields.
     *
     * @param lastFrame The last frame number to be set.
     */
    public void setLastFrame(int lastFrame) {
        view.setTfLastFrame(String.valueOf(lastFrame));
        settings.setLastFrame(String.valueOf(lastFrame));
        settings.setSlidingWindowLength(lastFrame / 20);
        updateStrideParamFields();
    }

    /**
     * Handles different bleach correction modes and initializes appropriate views for further user inputs
     * depending on the selected mode.
     *
     * @return ActionListener to handle bleach correction mode changes.
     */
    public ActionListener cbBleachCorChanged() {
        return (ActionEvent ev) -> {
            String bleachCorrectionMode = getComboBoxSelectionFromEvent(ev);
            settings.setBleachCorrection(bleachCorrectionMode);
            fitController.updateFitEnd(settings);

            if ("Sliding Window".equals(bleachCorrectionMode) || "Lin Segment".equals(bleachCorrectionMode)) {
                if (!imageController.isImageLoaded()) {
                    IJ.showMessage("No image open. Please open an image first.");
                    // reset the combo box to default in no image is loaded.
                    ((JComboBox<?>) ev.getSource()).setSelectedIndex(0);
                } else {
                    new SlidingWindowSelectionView(this::onBleachCorrectionSlidingWindowAccepted,
                            settings.getSlidingWindowLength());
                }
            } else if ("Polynomial".equals(bleachCorrectionMode)) {
                new PolynomialOrderSelectionView(this::onBleachCorrectionOrderAccepted,
                        bleachCorrectionModel.getPolynomialOrder());
            }
        };
    }

    /**
     * Callback method for accepting sliding window length.
     * It validates the input and sets it in the model or prompts re-entry if the input is invalid.
     *
     * @param slidingWindowLength The length of the sliding window chosen by the user.
     */
    private void onBleachCorrectionSlidingWindowAccepted(int slidingWindowLength) {
        int maxWindowSize = settings.getLastFrame() - settings.getFirstFrame();

        if (slidingWindowLength <= 0 || slidingWindowLength > maxWindowSize) {
            IJ.showMessage(
                    String.format("Invalid sliding window size. It must be inside 0 < order < %d", maxWindowSize));
            new SlidingWindowSelectionView(this::onBleachCorrectionSlidingWindowAccepted,
                    settings.getSlidingWindowLength());
        } else {
            settings.setSlidingWindowLength(slidingWindowLength);
            fitController.updateFitEnd(settings);
        }
    }

    /**
     * Callback method for accepting polynomial order.
     * It validates the polynomial order and sets it in the model or prompts re-entry if the input is invalid.
     *
     * @param order The polynomial order chosen by the user.
     */
    private void onBleachCorrectionOrderAccepted(int order) {
        if (order <= 0 || order > BleachCorrectionModel.MAX_POLYNOMIAL_ORDER) {
            IJ.showMessage(String.format("Invalid polynomial order. It must be inside 0 < order <= %d",
                    BleachCorrectionModel.MAX_POLYNOMIAL_ORDER));
            new PolynomialOrderSelectionView(this::onBleachCorrectionOrderAccepted,
                    bleachCorrectionModel.getPolynomialOrder());
        } else {
            bleachCorrectionModel.setPolynomialOrder(order);
        }
    }

    /**
     * Returns an action listener for changes in the filter selection combo box.
     * This method creates a listener that updates the filter settings and may
     * open a dialog for setting filter limits based on the selected filter mode.
     *
     * @return an ActionListener that processes filter selection changes
     */
    public ActionListener cbFilterChanged() {
        return (ActionEvent ev) -> {
            String filterMode = getComboBoxSelectionFromEvent(ev);
            settings.setFilter(filterMode);

            // If a filter mode other than "none" is selected, show the filter limits dialog.
            if (!filterMode.equals("none")) {
                new FilterLimitsSelectionView(this::onFilterSelectionAccepted, settings.getFilterLowerLimit(),
                        settings.getFilterUpperLimit());
            }
        };
    }

    /**
     * Creates an ActionListener that updates the fit model selection in the experimental settings model.
     * This listener is triggered when the fit model combo box selection changes.
     *
     * @return an ActionListener that processes the action event
     */
    public ActionListener cbFitModelChanged() {
        return (ActionEvent ev) -> {
            String fitModel = getComboBoxSelectionFromEvent(ev);
            settings.setFitModel(fitModel);
            updateSettingsField();
        };
    }

    /**
     * Callback method for accepting filter limits.
     * It validates the filter limits and sets them in the model or prompts re-entry if the input is invalid.
     *
     * @param lowerLimit The lower limit of the filter chosen by the user.
     * @param upperLimit The upper limit of the filter chosen by the user.
     */
    private void onFilterSelectionAccepted(int lowerLimit, int upperLimit) {
        if (lowerLimit < 0 || lowerLimit > upperLimit) {
            IJ.showMessage("Illegal filter limits");
            new FilterLimitsSelectionView(this::onFilterSelectionAccepted, settings.getFilterLowerLimit(),
                    settings.getFilterUpperLimit());
        } else {
            settings.setFilterLowerLimit(lowerLimit);
            settings.setFilterUpperLimit(upperLimit);
        }
    }

    /**
     * Returns an action listener to handle the exit button press event.
     * This listener disposes of the main panel view.
     *
     * @return an ActionListener that processes the exit button press event
     */
    public ActionListener btnExitPressed() {
        return (ActionEvent ev) -> view.dispose();
    }

    public ActionListener btnBtfPressed() {
        // TODO: FIXME
        return null;
    }

    /**
     * Returns an action listener to handle the exit button press event.
     * This listener disposes of the main panel view.
     *
     * @return an ActionListener that processes the exit button press event
     */
    public ActionListener btnUseExistingPressed() {
        return (ActionEvent ev) -> {
            if (WindowManager.getImageCount() > 0) {
                try {
                    imageController.loadImage(IJ.getImage());
                } catch (RuntimeException e) {
                    IJ.showMessage("Wrong image format", e.getMessage());
                }
            } else {
                IJ.showMessage("No image open.");
            }
        };
    }

    /**
     * Returns an action listener to handle the exit button press event.
     * This listener disposes of the main panel view.
     *
     * @return an ActionListener that processes the exit button press event
     */
    public ActionListener btnLoadNewPressed() {
        return (ActionEvent ev) -> {
            ImagePlus image = IJ.openImage();
            if (image != null) {
                try {
                    imageController.loadImage(image);
                } catch (RuntimeException e) {
                    IJ.showMessage("Wrong image format", e.getMessage());
                }
            }
        };
    }

    public ActionListener btnSavePressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnLoadPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnBatchPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnPSFPressed() {
        // TODO: FIXME
        return null;
    }

    public ItemListener tbDLPressed() {
        // TODO: FIXME
        return null;
    }

    /**
     * Returns an action listener to handle the "Options" button press event.
     * This listener opens the options controller.
     *
     * @return an ActionListener that processes the "Options" button press event
     */
    public ActionListener btnOptionsPressed() {
        return (ActionEvent ev) -> new OptionsController(optionsModel);
    }

    /**
     * Returns an action listener to handle the "Options" button press event.
     * This listener opens the options controller.
     *
     * @return an ActionListener that processes the "Options" button press event
     */
    public ItemListener tbNBPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setText(selected ? "N&B On" : "N&B Off");
            nbController.setVisible(selected);
        };
    }

    public ActionListener tbFilteringPressed() {
        // TODO: FIXME
        return null;
    }

    /**
     * Creates an ActionListener for the "Average" button.
     * This listener handles the calculation and plotting of the average correlation function
     * for the selected region of interest (ROI) in the image.
     *
     * @return The ActionListener instance.
     */
    public ActionListener btnAveragePressed() {
        return (ActionEvent ev) -> {
            if (!imageController.isImageLoaded()) {
                IJ.showMessage("No image open");
            } else if (correlator.getPixelsModel() == null) {
                IJ.showMessage("Nothing to plot, please run the correlation on at least one pixel before");
            } else {
                // if the ROI is null, we consider all correlated pixels.
                Roi roi = null;
                if (Plots.imgParam != null) {
                    roi = Plots.imgParam.getRoi();
                }

                PixelModel averagePixelModel =
                        AverageCorrelation.calculateAverageCorrelationFunction(correlator.getPixelsModel(), roi,
                                settings.getPixelBinning(), settings.getMinCursorPosition());
                fitController.fit(averagePixelModel, correlator.getLagTimes());

                if (optionsModel.isPlotACFCurves()) {
                    Plots.plotCorrelationFunction(Collections.singletonList(averagePixelModel),
                            correlator.getLagTimes(), null, settings.getBinning(), settings.getCCF(),
                            fitController.getFitStart(), fitController.getFitEnd());
                }

                if (settings.isMSD()) {
                    averagePixelModel.setMSD(
                            MeanSquareDisplacement.correlationToMSD(averagePixelModel.getAcf(), settings.getParamAx(),
                                    settings.getParamAy(), settings.getParamW(), settings.getSigmaZ(),
                                    settings.isMSD3d()));
                    Plots.plotMSD(Collections.singletonList(averagePixelModel), correlator.getLagTimes(), null,
                            settings.getBinning());
                }
            }
        };
    }

    /**
     * Creates an ActionListener for the button press event that generates a scatter plot.
     *
     * @return An ActionListener that handles the button press event.
     */
    public ActionListener btnParaCorPressed() {
        return (ActionEvent ev) -> {
            if (!imageController.isImageLoaded()) {
                IJ.showMessage("No image open.");
            } else if (correlator.getPixelsModel() == null) {
                IJ.showMessage("Nothing to plot, please run the fit on at least one pixel before.");
            } else {
                Pair<double[][], String[]> scatterArrayAndLabels =
                        PixelModel.getScatterPlotArray(correlator.getPixelsModel(), settings.getParaCor());
                double[][] scPlot = scatterArrayAndLabels.getLeft();
                String[] labels = scatterArrayAndLabels.getRight();

                Plots.scatterPlot(scPlot, labels[0], labels[1]);
            }
        };
    }

    /**
     * Creates an ActionListener that handles the DCCF button press event.
     *
     * @return An ActionListener that initiates the DCCF computation when the button is pressed.
     */
    public ActionListener btnDCCFPressed() {
        return (ActionEvent ev) -> {
            if (!imageController.isImageLoaded()) {
                IJ.showMessage("No image open.");
            } else {
                String directionName = settings.getdCCF();
                IJ.showStatus("Correlating all pixels");
                DeltaCCFWorker dccfWorker =
                        new DeltaCCFWorker(settings, correlator, imageController.getImage(), directionName,
                                (dccfArray, direction) -> {
                                    IJ.showStatus("Done");
                                    Plots.plotDCCFWindow(dccfArray, direction);
                                });
                dccfWorker.execute();
            }
        };
    }

    public ActionListener btnRTPressed() {
        // TODO: FIXME
        return null;
    }

    /**
     * Returns an ActionListener that writes the current configuration to a YAML file.
     * The configuration includes settings and options which are serialized and saved to the file path
     * specified by getConfigPath. If an error occurs during writing, a message is displayed.
     *
     * @return an ActionListener that writes the configuration to a file when triggered.
     */
    public ActionListener btnWriteConfigPressed() {
        return (ActionEvent ev) -> {
            Map<String, Object> data = new HashMap<>();
            data.put("Settings", settings.toMap());
            data.put("Options", optionsModel.toMap());

            DumperOptions optionsYaml = new DumperOptions();
            optionsYaml.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
            optionsYaml.setPrettyFlow(true);

            Yaml yaml = new Yaml(optionsYaml);
            try (FileWriter writer = new FileWriter(getConfigPath())) {
                yaml.dump(data, writer);
            } catch (IOException e) {
                IJ.showMessage("Error", "Can't write configuration.");
            }
        };
    }

    public ActionListener btnDCRPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnParamVideoPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnDebugPressed() {
        // TODO: FIXME
        return null;
    }

    /**
     * Creates an ActionListener that handles the event when the ROI button is pressed.
     * It checks if an image is loaded and if an ROI is selected. If a CCF distance is set, it creates
     * an additional ROI for correlation, validates it, and performs correlation.
     *
     * @return An ActionListener that processes the ROI selection and performs correlation.
     */
    public ActionListener btnROIPressed() {
        return (ActionEvent ev) -> {
            if (!imageController.isImageLoaded()) {
                IJ.showMessage("No image open.");
                return;
            }

            Roi imgRoi = imageController.getImage().getRoi();

            if (imgRoi == null) {
                IJ.showMessage("No ROI chosen.");
            } else {
                // create another ROI if the distance is not null
                if (settings.getCCF().width != 0 || settings.getCCF().height != 0) {
                    Roi CCFRoi = (Roi) imgRoi.clone();
                    CCFRoi.setLocation(imgRoi.getBounds().getX() + settings.getCCF().width,
                            imgRoi.getBounds().getY() + settings.getCCF().height);
                    CCFRoi.setStrokeColor(Color.RED);
                    if (!imageController.isROIValid(CCFRoi)) {
                        IJ.showMessage("Correlation points are not within image.");
                        return;
                    }
                    imageController.getImage().setOverlay(new Overlay(CCFRoi));
                }

                // Perform ROI
                IJ.showStatus("Correlating pixels");
                ROIWorker worker = new ROIWorker(() -> imageController.correlateROI(imgRoi));
                worker.execute();
            }
        };
    }

    /**
     * Creates an ActionListener that handles the event when the "All" button is pressed.
     * It checks if an image is loaded and then sets the ROI to cover the entire image,
     * adjusted for CCF and binning settings, before performing correlation.
     *
     * @return An ActionListener that sets the ROI and performs correlation.
     */
    public ActionListener btnAllPressed() {
        return (ActionEvent ev) -> {
            if (!imageController.isImageLoaded()) {
                IJ.showMessage("No image open.");
                return;
            }

            Overlay overlay = imageController.getImage().getOverlay();
            if (overlay != null) {
                overlay.clear();
            }

            Point startLocation = settings.getMinCursorPosition();
            Point endLocation = settings.getMaxCursorPosition(imageController.getImage());
            Point pixelBinning = settings.getPixelBinning();

            int startX = startLocation.x * pixelBinning.x;
            int width = (endLocation.x - startLocation.x + 1) * pixelBinning.x;
            int startY = startLocation.y * pixelBinning.y;
            int height = (endLocation.y - startLocation.y + 1) * pixelBinning.y;

            Roi imgRoi = new Roi(startX, startY, width, height);
            imageController.getImage().setRoi(imgRoi);

            btnROIPressed().actionPerformed(ev);
        };
    }

    public ItemListener tbFCCSDisplayPressed() {
        // TODO: FIXME
        return null;
    }

    /**
     * Returns an item listener to handle the "Exp Settings" toggle button press event.
     * This listener toggles the visibility of the experimental settings view.
     *
     * @return an ItemListener that processes the "Exp Settings" toggle button press event
     */
    public ItemListener tbExpSettingsPressed() {
        return (ItemEvent ev) -> expSettingsView.setVisible(ev.getStateChange() == ItemEvent.SELECTED);
    }

    /**
     * Returns an item listener to handle the "Bleach Correction" toggle button press event.
     * This listener toggles the visibility of the bleach correction view, ensuring an image is loaded first.
     *
     * @return an ItemListener that processes the "Bleach Correction" toggle button press event
     */
    public ItemListener tbBleachCorStridePressed() {
        return (ItemEvent ev) -> {
            if (imageController.isImageLoaded()) {
                bleachCorrectionView.setVisible(ev.getStateChange() == ItemEvent.SELECTED);
            } else if (ev.getStateChange() == ItemEvent.SELECTED) {
                // Don't show the message if the button is unselected.
                IJ.showMessage("No image stack loaded.");
                ((JToggleButton) ev.getSource()).setSelected(false);
            }
        };
    }

    /**
     * Returns an item listener to handle the "Fit" toggle button press event.
     * This listener toggles the visibility of the FitController.
     *
     * @return an ItemListener that processes the "Fit" toggle button press event
     */
    public ItemListener tbFitPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setText(selected ? "Fit On" : "Fit Off");
            fitController.setVisible(selected);
        };
    }

    /**
     * Returns an item listener to handle the "Sim" toggle button press event.
     * This listener toggles the visibility of the SimulationController.
     *
     * @return an ItemListener that processes the "Sim" toggle button press event
     */
    public ItemListener tbSimPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setText(selected ? "Sim On" : "Sim Off");
            simulationController.setVisible(selected);
        };
    }

    /**
     * Creates an ItemListener for the MSD toggle button.
     * When the button is pressed, it toggles the MSD (Mean Squared Displacement) analysis on or off
     * and updates the button text accordingly. If MSD is enabled, it opens an MSDView dialog to
     * configure the MSD settings.
     *
     * @return an ItemListener for the MSD toggle button.
     */
    public ItemListener tbMSDPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);

            if (selected) {
                settings.setMSD(true);
                button.setText("MSD On");
                new MSDView(settings.isMSD3d(), settings::setMSD3d);
            } else {
                settings.setMSD(false);
                button.setText("MSD Off");
            }
        };
    }

    /**
     * Returns an item listener to handle the "Overlap" toggle button press event.
     * This listener toggles the overlap setting in the experimental settings model.
     *
     * @return an ItemListener that processes the "Overlap" toggle button press event
     */
    public ItemListener tbOverlapPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();
            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);

            button.setText(selected ? "Overlap On" : "Overlap Off");
            settings.setOverlap(selected);
        };
    }

    /**
     * Returns an item listener to handle the "Background" toggle button press event.
     * This listener toggles the visibility of the BackgroundSubtractionController.
     *
     * @return an ItemListener that processes the "Background" toggle button press event
     */
    public ItemListener tbBackgroundPressed() {
        return (ItemEvent ev) -> {
            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            backgroundSubtractionController.setVisible(selected);
        };
    }

    /**
     * Updates the display fields in the experimental settings view based on the current values in the model.
     * This method ensures that the UI reflects the most up-to-date settings.
     */
    private void updateSettingsField() {
        Runnable doUpdateSettingsField = () -> {
            settings.updateSettings();
            expSettingsView.setNonUserSettings();
        };

        // Execute the update in the Swing event dispatch thread to ensure thread safety
        SwingUtilities.invokeLater(doUpdateSettingsField);
    }

    /**
     * Decorates a setter from the model to automatically update the settings view
     * after the model value has been changed.
     *
     * @param setter A Consumer that sets a value in the model.
     * @return A FocusListener that updates the model and then the view when focus is lost.
     */
    public FocusListener updateSettings(Consumer<String> setter) {
        // decorate the setter to call updateSettingsfield after changing the value
        Consumer<String> decoratedSetter = (String value) -> {
            setter.accept(value);
            updateSettingsField();
        };

        // Create the focus listener with the method from the factory with the decorated setter
        return createFocusListener(decoratedSetter);
    }

    /**
     * Updates the stride parameter fields based on the current model values.
     * This method ensures that the stride parameter fields are accurately set
     * based on the number of frames and the average stride.
     */
    private void updateStrideParamFields() {
        Runnable doUpdateStrideParam = () -> {
            int numberOfFrames = settings.getLastFrame() - settings.getFirstFrame() + 1;

            // Use variable points for the intensity, except when less than 1000 frames are present.
            int numPointsIntensityTrace = numberOfFrames;
            if (numberOfFrames >= 1000) {
                numPointsIntensityTrace = numberOfFrames / bleachCorrectionModel.getAverageStride();
            }

            bleachCorrectionView.setTextNumPointsIntensityTrace(String.valueOf(numPointsIntensityTrace));
            bleachCorrectionModel.setNumPointsIntensityTrace(numPointsIntensityTrace);
        };

        SwingUtilities.invokeLater(doUpdateStrideParam);
    }

    /**
     * Decorates a setter from the model to automatically update the stride parameter fields
     * after the model value has been changed.
     *
     * @param setter A Consumer that sets a value in the model.
     * @return A FocusListener that updates the model and then the view when focus is lost.
     */
    public FocusListener updateStrideParam(Consumer<String> setter) {
        Consumer<String> decoratedSetter = (String value) -> {
            setter.accept(value);
            updateStrideParamFields();
        };

        return createFocusListener(decoratedSetter);
    }

    /**
     * Returns a Consumer that decorates the given setter to also update the fit end.
     *
     * @param setter the original setter Consumer
     * @return a decorated Consumer that updates the fit end and calls the given setter
     */
    public Consumer<String> updateFitEnd(Consumer<String> setter) {
        return (String value) -> {
            setter.accept(value);
            fitController.updateFitEnd(settings);
        };
    }
}