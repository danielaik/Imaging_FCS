package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.HardwareModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.OptionsModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.BleachCorrelationView;
import fiji.plugin.imaging_fcs.new_imfcs.view.ExpSettingsView;
import fiji.plugin.imaging_fcs.new_imfcs.view.FilterLimitsSelectionView;
import fiji.plugin.imaging_fcs.new_imfcs.view.MainPanelView;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;

import javax.swing.*;
import java.awt.event.*;
import java.util.function.Consumer;
import java.util.function.Function;

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
    private final BleachCorrelationView bleachCorrelationView;
    private final HardwareModel hardwareModel;
    private final OptionsModel optionsModel;
    private final ImageController imageController;
    private final ExpSettingsModel expSettingsModel;
    private final SimulationController simulationController;
    private final BackgroundSubtractionController backgroundSubtractionController;
    private final NBController nbController;

    /**
     * Constructor that initializes models, views, and other controllers needed for the main panel.
     *
     * @param hardwareModel The model containing hardware settings for the imaging FCS analysis.
     */
    public MainPanelController(HardwareModel hardwareModel) {
        this.hardwareModel = hardwareModel;
        this.optionsModel = new OptionsModel(hardwareModel.isCuda());

        ImageModel imageModel = new ImageModel();
        this.imageController = new ImageController(this, imageModel);

        this.expSettingsModel = new ExpSettingsModel();
        this.expSettingsView = new ExpSettingsView(this, expSettingsModel);
        updateSettingsField();

        this.bleachCorrelationView = new BleachCorrelationView(this, expSettingsModel);

        this.simulationController = new SimulationController(imageController, expSettingsModel);

        this.backgroundSubtractionController = new BackgroundSubtractionController(imageModel);

        this.nbController = new NBController(imageModel, expSettingsModel, optionsModel);

        this.view = new MainPanelView(this, this.expSettingsModel);
    }

    public void setLastFrame(int lastFrame) {
        view.setTfLastFrame(String.valueOf(lastFrame));
    }

    public ActionListener cbBleachCorChanged() {
        // TODO: FIXME
        return null;
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
            String filterMode = ControllerUtils.getComboBoxSelectionFromEvent(ev);
            expSettingsModel.setFilter(filterMode);

            // If a filter mode other than "none" is selected, show the filter limits dialog.
            if (!filterMode.equals("none")) {
                new FilterLimitsSelectionView(this::onFilterSelectionAccepted, expSettingsModel.getFilterLowerLimit(),
                        expSettingsModel.getFilterUpperLimit());
            }
        };
    }

    /**
     * Handles the acceptance of filter limits from the FilterLimitsSelectionView dialog.
     * Validates the entered limits and updates the model accordingly or prompts re-entry if limits are invalid.
     *
     * @param filterView The FilterLimitsSelectionView instance from which to retrieve the limits
     */
    private void onFilterSelectionAccepted(FilterLimitsSelectionView filterView) {
        int lowerLimit = (int) filterView.getNextNumber();
        int upperLimit = (int) filterView.getNextNumber();

        if (lowerLimit < 0 || lowerLimit > upperLimit) {
            IJ.showMessage("Illegal filter limits");
            new FilterLimitsSelectionView(this::onFilterSelectionAccepted, expSettingsModel.getFilterLowerLimit(),
                    expSettingsModel.getFilterUpperLimit());
        } else {
            expSettingsModel.setFilterLowerLimit(lowerLimit);
            expSettingsModel.setFilterUpperLimit(upperLimit);
        }
    }

    public ActionListener btnExitPressed() {
        return (ActionEvent ev) -> view.dispose();
    }

    public ActionListener btnBtfPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnUseExistingPressed() {
        return (ActionEvent ev) -> {
            if (WindowManager.getImageCount() > 0) {
                try {
                    imageController.loadImage(IJ.getImage(), false);
                } catch (RuntimeException e) {
                    IJ.showMessage("Wrong image format", e.getMessage());
                }
            } else {
                IJ.showMessage("No image open.");
            }
        };
    }

    public ActionListener btnLoadNewPressed() {
        return (ActionEvent ev) -> {
            ImagePlus image = IJ.openImage();
            if (image != null) {
                try {
                    imageController.loadImage(image, false);
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

    public ActionListener btnOptionsPressed() {
        return (ActionEvent ev) -> new OptionsController(optionsModel);
    }

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

    public ActionListener btnAvePressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnParaCorPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnDCCFPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnRTPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnWriteConfigPressed() {
        // TODO: FIXME
        return null;
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

    public ActionListener btnROIPressed() {
        // TODO: FIXME
        return null;
    }

    public ActionListener btnAllPressed() {
        // TODO: FIXME
        return null;
    }

    public ItemListener tbFCCSDisplayPressed() {
        // TODO: FIXME
        return null;
    }

    public ItemListener tbExpSettingsPressed() {
        return (ItemEvent ev) -> expSettingsView.setVisible(ev.getStateChange() == ItemEvent.SELECTED);
    }

    public ItemListener tbBleachCorStridePressed() {
        return (ItemEvent ev) -> {
            if (imageController.isImageLoaded()) {
                bleachCorrelationView.setVisible(ev.getStateChange() == ItemEvent.SELECTED);
            } else if (ev.getStateChange() == ItemEvent.SELECTED) {
                // Don't show the message if the button is unselected.
                IJ.showMessage("No image stack loaded.");
                ((JToggleButton) ev.getSource()).setSelected(false);
            }
        };
    }

    public ItemListener tbFitPressed() {
        // TODO: FIXME
        return null;
    }

    public ItemListener tbSimPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setText(selected ? "Sim On" : "Sim Off");
            simulationController.setVisible(selected);
        };
    }

    public ActionListener tbMSDPressed() {
        // TODO: FIXME
        return null;
    }

    public ItemListener tbOverlapPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();
            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);

            button.setText(selected ? "Overlap On" : "Overlap Off");
            expSettingsModel.setOverlap(selected);
        };
    }

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
            expSettingsModel.updateSettings();

            // Use scientific notation for these fields
            Function<Double, String> formatter = value -> String.format("%6.2e", value);

            expSettingsView.setTextParamA(formatter.apply(expSettingsModel.getParamA()));
            expSettingsView.setTextParamW(formatter.apply(expSettingsModel.getParamW()));
            expSettingsView.setTextParamW2(formatter.apply(expSettingsModel.getParamW2()));
            expSettingsView.setTextParamZ(formatter.apply(expSettingsModel.getParamZ()));
            expSettingsView.setTextParamZ2(formatter.apply(expSettingsModel.getParamZ2()));
            expSettingsView.setTextParamRx(formatter.apply(expSettingsModel.getParamRx()));
            expSettingsView.setTextParamRy(formatter.apply(expSettingsModel.getParamRy()));
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

    private void updateStrideParamFields() {
        Runnable doUpdateStrideParam = () -> {
            int numberOfFrames = expSettingsModel.getLastFrame() - expSettingsModel.getFirstFrame() + 1;

            // Use variable points for the intensity, except when less than 1000 frames are present.
            int numPointsIntensityTrace = numberOfFrames;
            if (numberOfFrames >= 1000) {
                numPointsIntensityTrace = numberOfFrames / expSettingsModel.getAverageStride();
            }

            bleachCorrelationView.setTextNumPointsIntensityTrace(String.valueOf(numPointsIntensityTrace));
        };

        SwingUtilities.invokeLater(doUpdateStrideParam);
    }

    public FocusListener updateStrideParam(Consumer<String> setter) {
        Consumer<String> decoratedSetter = (String value) -> {
            setter.accept(value);
            updateStrideParamFields();
        };

        return createFocusListener(decoratedSetter);
    }
}