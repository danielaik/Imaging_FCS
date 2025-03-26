package fiji.plugin.imaging_fcs.imfcs.view;

import fiji.plugin.imaging_fcs.imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.imfcs.controller.FilteringController;
import fiji.plugin.imaging_fcs.imfcs.model.FilteringModel;
import fiji.plugin.imaging_fcs.imfcs.model.FitModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionListener;
import java.util.function.Function;

import static fiji.plugin.imaging_fcs.imfcs.controller.FieldListenerFactory.createFocusListener;
import static fiji.plugin.imaging_fcs.imfcs.view.ButtonFactory.createJButton;
import static fiji.plugin.imaging_fcs.imfcs.view.TextFieldFactory.createTextField;
import static fiji.plugin.imaging_fcs.imfcs.view.TextFieldFactory.setText;
import static fiji.plugin.imaging_fcs.imfcs.view.UIUtils.createJLabel;

/**
 * The FilteringView class is responsible for creating and managing the UI
 * components related to threshold settings for filtering parameter maps
 * in a fluorescence correlation spectroscopy (FCS) fitting analysis.
 */
public class FilteringView extends BaseView {
    // TODO: add columns for DCCF and two row for rel G and chi2
    private static final GridLayout FILTERING_LAYOUT = new GridLayout(17, 6); // should be 17 and 6
    private static final Point FILTERING_POSITION =
            new Point(Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 310,
                    Constants.MAIN_PANEL_POS.y + 335);
    private static final Dimension FILTERING_DIMENSION = new Dimension(420, 300);
    private final FilteringController controller;

    // The model that holds the data and thresholds to be displayed and manipulated in this view
    private final FitModel model;

    // UI components for text fields and radio buttons associated with each parameter
    private FilterFields filterN, filterD, filterVx, filterVy, filterG, filterF2, filterD2, filterF3, filterD3,
            filterFTrip, filterTTrip, filterChi2;
    private JRadioButton rbtnSameAsCCF;
    private JButton btnFilter, btnReset, btnLoadBinaryFilter;

    /**
     * Constructs the {@code FilteringView} with the specified {@code FilteringController} and {@code FitModel}.
     *
     * @param controller the controller managing the filtering actions
     * @param model      the model containing threshold and parameter data
     */
    public FilteringView(FilteringController controller, FitModel model) {
        super("Thresholds settings");
        this.controller = controller;
        this.model = model;

        initializeUI();
    }

    @Override
    protected void configureWindow() {
        super.configureWindow();

        setLayout(FILTERING_LAYOUT);
        setLocation(FILTERING_POSITION);
        setSize(FILTERING_DIMENSION);

        setVisible(false);
    }

    @Override
    protected void initializeTextFields() {
        filterN = new FilterFields(model.getN().getThreshold(), controller::enabledThresholdPressed);
        filterD = new FilterFields(model.getD().getThreshold(), controller::enabledThresholdPressed);
        filterVx = new FilterFields(model.getVx().getThreshold(), controller::enabledThresholdPressed);
        filterVy = new FilterFields(model.getVy().getThreshold(), controller::enabledThresholdPressed);
        filterG = new FilterFields(model.getG().getThreshold(), controller::enabledThresholdPressed);
        filterF2 = new FilterFields(model.getF2().getThreshold(), controller::enabledThresholdPressed);
        filterD2 = new FilterFields(model.getD2().getThreshold(), controller::enabledThresholdPressed);
        filterF3 = new FilterFields(model.getF3().getThreshold(), controller::enabledThresholdPressed);
        filterD3 = new FilterFields(model.getD3().getThreshold(), controller::enabledThresholdPressed);
        filterFTrip = new FilterFields(model.getFTrip().getThreshold(), controller::enabledThresholdPressed);
        filterTTrip = new FilterFields(model.getTTrip().getThreshold(), controller::enabledThresholdPressed);
        filterChi2 = new FilterFields(model.getChi2Threshold(), controller::enabledThresholdPressed);
    }

    @Override
    protected void initializeButtons() {
        rbtnSameAsCCF = new JRadioButton();
        rbtnSameAsCCF.setEnabled(false);
        rbtnSameAsCCF.addActionListener(controller.sameAsCCFPressed());

        btnFilter = createJButton("Filter",
                "Creates filtering mask according to specified thresholds and applies it on the parameter maps", null,
                controller.btnFilteringPressed());
        btnReset = createJButton("Reset",
                "Resets the thresholds to their default values and the filtering mask to 1.0 for all fitted pixels",
                null, controller.btnResetPressed());
        btnLoadBinaryFilter = createJButton("Binary", "", null, controller.btnLoadBinaryFilterPressed());
    }

    @Override
    protected void addComponentsToFrame() {
        // row 1
        add(btnLoadBinaryFilter);
        add(createJLabel(" ", ""));
        add(createJLabel("CF", ""));
        add(createJLabel(" ", ""));
        add(createJLabel("ACF 1/2", ""));
        add(createJLabel(" ", ""));

        // row 2
        add(createJLabel(" ", ""));
        add(createJLabel("filter", ""));
        add(createJLabel("Min.", ""));
        add(createJLabel("Max.", ""));
        add(createJLabel("Min.", ""));
        add(createJLabel("Max.", ""));

        // row 3
        add(createJLabel("N", ""));
        add(filterN);

        // row 4
        add(createJLabel("D [um2/s]", ""));
        add(filterD);

        // row 5
        add(createJLabel("vx [um/s]", ""));
        add(filterVx);

        // row 6
        add(createJLabel("vy [um/s]", ""));
        add(filterVy);

        // row 7
        add(createJLabel("G", ""));
        add(filterG);

        // TODO: add rel.G

        // row 8
        add(createJLabel("F2", ""));
        add(filterF2);

        // row 9
        add(createJLabel("D2 [um2/s]", ""));
        add(filterD2);

        // row 10
        add(createJLabel("F3", ""));
        add(filterF3);

        // row 11
        add(createJLabel("D3 [um2/s]", ""));
        add(filterD3);

        // row 12
        add(createJLabel("Ftrip", ""));
        add(filterFTrip);

        // row 13
        add(createJLabel("Ttrip", ""));
        add(filterTTrip);

        // row 14
        add(createJLabel("Chi2", ""));
        add(filterChi2);

        // row 15
        add(btnFilter);
        add(createJLabel(" ", ""));
        add(createJLabel("ACF 1/2 ", ""));
        add(createJLabel("same as ", ""));
        add(createJLabel("CCF", ""));
        add(rbtnSameAsCCF);

        // row 16
        add(btnReset);
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
    }

    /**
     * Refreshes the filter fields based on the current model state.
     */
    public void refreshFields() {
        for (FilterFields filterFields : new FilterFields[]{
                filterN,
                filterD,
                filterVx,
                filterVy,
                filterG,
                filterF2,
                filterD2,
                filterF3,
                filterD3,
                filterFTrip,
                filterTTrip,
                filterChi2
        }) {
            filterFields.refresh();
        }

        if (FilteringModel.getFilteringBinaryImage() != null) {
            btnLoadBinaryFilter.setText("Loaded");
        } else {
            btnLoadBinaryFilter.setText("Binary");
        }
    }

    /**
     * Adds the specified {@code FilterFields} components (radio button, min/max text fields) to the frame.
     *
     * @param filterFields the {@code FilterFields} containing the UI components to be added
     */
    private void add(FilterFields filterFields) {
        add(filterFields.radioButton);
        add(filterFields.min);
        add(filterFields.max);
        add(filterFields.minAcf);
        add(filterFields.maxAcf);
    }

    /**
     * Enables or disables the "Same as CCF" button based on the specified value.
     *
     * @param b whether to enable the button
     */
    public void enableButtonSameAsCCF(boolean b) {
        if (!b) {
            // unselect the button if it was selected when we deactivate the button.
            rbtnSameAsCCF.setSelected(false);
        }
        rbtnSameAsCCF.setEnabled(b);
    }

    /**
     * Inner class representing the filter text fields (min and max) for each parameter.
     * Each parameter has a radio button for enabling/disabling the threshold, and
     * text fields for setting the min/max thresholds for both CF and ACF.
     */
    public static class FilterFields {
        private final FilteringModel threshold;
        private final JRadioButton radioButton;
        private final JTextField min;
        private final JTextField max;
        private final JTextField minAcf;
        private final JTextField maxAcf;

        /**
         * Constructs {@code FilterFields} for a given threshold and listener.
         *
         * @param threshold               the threshold model linked to the fields
         * @param enabledThresholdPressed the action listener for enabling/disabling the threshold
         */
        public FilterFields(FilteringModel threshold, Function<FilterFields, ActionListener> enabledThresholdPressed) {
            this.threshold = threshold;
            this.radioButton = createRadioButton(enabledThresholdPressed);
            FilteringModel acfThreshold = threshold.getAcfThreshold();

            min = createTextField(threshold.getMin(), "", createFocusListener(threshold::setMin));
            max = createTextField(threshold.getMax(), "", createFocusListener(threshold::setMax));
            minAcf = createTextField(acfThreshold.getMin(), "", createFocusListener(acfThreshold::setMin));
            maxAcf = createTextField(acfThreshold.getMax(), "", createFocusListener(acfThreshold::setMax));

            this.refreshEnabled();
        }

        /**
         * Formats a given double value to two decimal places.
         *
         * @param value The double value to be formatted.
         * @return A string representation of the value rounded to two decimal places.
         */
        private static String formatValue(double value) {
            return String.format("%.2f", value);
        }

        /**
         * Creates a radio button with an action listener to enable or disable the threshold.
         *
         * @param enabledThresholdPressed the action listener for the radio button
         * @return the created {@code JRadioButton}
         */
        private JRadioButton createRadioButton(Function<FilterFields, ActionListener> enabledThresholdPressed) {
            JRadioButton radioButton = new JRadioButton();
            radioButton.setSelected(threshold.getActive());

            radioButton.addActionListener(enabledThresholdPressed.apply(this));

            return radioButton;
        }

        /**
         * Refreshes the text fields and radio button based on the current threshold values.
         * Updates the UI elements with the current min/max values for both CF and ACF.
         */
        public void refresh() {
            setText(min, formatValue(threshold.getMin()));
            setText(max, formatValue(threshold.getMax()));
            setText(minAcf, formatValue(threshold.getAcfThreshold().getMin()));
            setText(maxAcf, formatValue(threshold.getAcfThreshold().getMax()));
            refreshEnabled();
        }

        /**
         * Enables or disables the text fields based on whether the threshold is active.
         * If the threshold is active, the corresponding text fields are enabled for editing.
         */
        public void refreshEnabled() {
            radioButton.setSelected(threshold.getActive());
            min.setEnabled(threshold.getActive());
            max.setEnabled(threshold.getActive());
            minAcf.setEnabled(threshold.getAcfActive());
            maxAcf.setEnabled(threshold.getAcfActive());
        }

        public FilteringModel getThreshold() {
            return threshold;
        }
    }
}
