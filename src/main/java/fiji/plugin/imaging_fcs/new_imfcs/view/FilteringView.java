package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.FilteringController;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.FieldListenerFactory.createFocusListener;
import static fiji.plugin.imaging_fcs.new_imfcs.view.ButtonFactory.createJButton;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.createTextField;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.setText;
import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

/**
 * The FilteringView class is responsible for creating and managing the UI
 * components related to threshold settings for filtering parameter maps
 * in a fluorescence correlation spectroscopy (FCS) fitting analysis.
 */
public class FilteringView extends BaseView {
    // TODO: add columns for DCCF and two row for rel G and chi2
    private static final GridLayout FILTERING_LAYOUT = new GridLayout(17, 4); // should be 17 and 6
    private static final Point FILTERING_POSITION =
            new Point(Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 310,
                    Constants.MAIN_PANEL_POS.y + 335);
    private static final Dimension FILTERING_DIMENSION = new Dimension(420, 300);
    private final FilteringController controller;

    // The model that holds the data and thresholds to be displayed and manipulated in this view
    private final FitModel model;

    // UI components for text fields and radio buttons associated with each parameter
    private FilterTextField filterN, filterD, filterVx, filterVy, filterG, filterF2, filterD2, filterF3, filterD3,
            filterFTrip, filterTTrip, filterChi2;
    private JRadioButton rbtnFilterN, rbtnFilterD, rbtnFilterVx, rbtnFilterVy, rbtnFilterG, rbtnFilterF2, rbtnFilterD2,
            rbtnFilterF3, rbtnFilterD3, rbtnFilterFTrip, rbtnFilterTTrip, rbtnFilterChi2;
    private JButton btnFilter, btnReset, btnLoadBinaryFilter;

    /**
     * Constructs the FilteringView with the specified FitModel.
     *
     * @param model the model containing the thresholds and parameter data
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
        filterN = new FilterTextField(model.getN().getThreshold());
        filterD = new FilterTextField(model.getD().getThreshold());
        filterVx = new FilterTextField(model.getVx().getThreshold());
        filterVy = new FilterTextField(model.getVy().getThreshold());
        filterG = new FilterTextField(model.getG().getThreshold());
        filterF2 = new FilterTextField(model.getF2().getThreshold());
        filterD2 = new FilterTextField(model.getD2().getThreshold());
        filterF3 = new FilterTextField(model.getF3().getThreshold());
        filterD3 = new FilterTextField(model.getD3().getThreshold());
        filterFTrip = new FilterTextField(model.getFTrip().getThreshold());
        filterTTrip = new FilterTextField(model.getTTrip().getThreshold());
        filterChi2 = new FilterTextField(model.getChi2Threshold());
    }

    /**
     * Resets all filter fields and corresponding radio buttons in the view.
     * <p>
     * This method clears the values in each filter field and unselects the
     * associated radio buttons, returning the view to its default state.
     */
    public void resetFields() {
        filterN.reset();
        filterD.reset();
        filterVx.reset();
        filterVy.reset();
        filterG.reset();
        filterF2.reset();
        filterD2.reset();
        filterF3.reset();
        filterD3.reset();
        filterFTrip.reset();
        filterTTrip.reset();
        filterChi2.reset();

        rbtnFilterN.setSelected(false);
        rbtnFilterD.setSelected(false);
        rbtnFilterVx.setSelected(false);
        rbtnFilterVy.setSelected(false);
        rbtnFilterG.setSelected(false);
        rbtnFilterF2.setSelected(false);
        rbtnFilterD2.setSelected(false);
        rbtnFilterF3.setSelected(false);
        rbtnFilterD3.setSelected(false);
        rbtnFilterFTrip.setSelected(false);
        rbtnFilterTTrip.setSelected(false);
        rbtnFilterChi2.setSelected(false);

        btnLoadBinaryFilter.setText("Binary");
    }

    /**
     * Creates a JRadioButton linked to a specific threshold and associated text field.
     *
     * @param filterTextField the text field that will be enabled or disabled based on the radio button state
     * @return the created JRadioButton
     */
    private JRadioButton createRadioButton(FilterTextField filterTextField) {
        JRadioButton radioButton = new JRadioButton();
        FitModel.Threshold threshold = filterTextField.threshold;
        radioButton.setSelected(threshold.getActive());

        radioButton.addActionListener(ev -> {
            threshold.setActive(radioButton.isSelected());
            filterTextField.setEnabled(radioButton.isSelected());
        });

        return radioButton;
    }

    /**
     * Initializes the radio buttons for each filter parameter.
     */
    private void initializeRadioButtons() {
        rbtnFilterN = createRadioButton(filterN);
        rbtnFilterD = createRadioButton(filterD);
        rbtnFilterVx = createRadioButton(filterVx);
        rbtnFilterVy = createRadioButton(filterVy);
        rbtnFilterG = createRadioButton(filterG);
        rbtnFilterF2 = createRadioButton(filterF2);
        rbtnFilterD2 = createRadioButton(filterD2);
        rbtnFilterF3 = createRadioButton(filterF3);
        rbtnFilterD3 = createRadioButton(filterD3);
        rbtnFilterFTrip = createRadioButton(filterFTrip);
        rbtnFilterTTrip = createRadioButton(filterTTrip);
        rbtnFilterChi2 = createRadioButton(filterChi2);
    }

    @Override
    protected void initializeButtons() {
        initializeRadioButtons();

        btnFilter = createJButton("Filter",
                "Creates filtering mask according to specified thresholds and applies it on the parameter maps", null,
                controller.btnFilteringPressed());
        btnReset = createJButton("Reset",
                "Resets the thresholds to their default values and the filtering mask to 1.0 for all fitted pixels",
                null, controller.btnResetPressed());
        btnLoadBinaryFilter = createJButton("Binary", "", null, controller.btnLoadBinaryFilterPressed());
    }

    /**
     * Adds the components (text fields for min and max values) to the frame.
     *
     * @param filterTextField the FilterTextField containing the min and max text fields
     */
    private void add(FilterTextField filterTextField) {
        add(filterTextField.min);
        add(filterTextField.max);
    }

    @Override
    protected void addComponentsToFrame() {
        // row 1
        add(btnLoadBinaryFilter);
        add(createJLabel(" ", ""));
        add(createJLabel("ACF", ""));
        add(createJLabel(" ", ""));

        // row 2
        add(createJLabel(" ", ""));
        add(createJLabel("filter", ""));
        add(createJLabel("Min.", ""));
        add(createJLabel("Max.", ""));

        // row 3
        add(createJLabel("N", ""));
        add(rbtnFilterN);
        add(filterN);

        // row 4
        add(createJLabel("D [um2/s]", ""));
        add(rbtnFilterD);
        add(filterD);

        // row 5
        add(createJLabel("vx [um/s]", ""));
        add(rbtnFilterVx);
        add(filterVx);

        // row 6
        add(createJLabel("vy [um/s]", ""));
        add(rbtnFilterVy);
        add(filterVy);

        // row 7
        add(createJLabel("G", ""));
        add(rbtnFilterG);
        add(filterG);

        // TODO: add rel.G

        // row 8
        add(createJLabel("F2", ""));
        add(rbtnFilterF2);
        add(filterF2);

        // row 9
        add(createJLabel("D2 [um2/s]", ""));
        add(rbtnFilterD2);
        add(filterD2);

        // row 10
        add(createJLabel("F3", ""));
        add(rbtnFilterF3);
        add(filterF3);

        // row 11
        add(createJLabel("D3 [um2/s]", ""));
        add(rbtnFilterD3);
        add(filterD3);

        // row 12
        add(createJLabel("Ftrip", ""));
        add(rbtnFilterFTrip);
        add(filterFTrip);

        // row 13
        add(createJLabel("Ttrip", ""));
        add(rbtnFilterTTrip);
        add(filterTTrip);

        // row 14
        add(createJLabel("Chi2", ""));
        add(rbtnFilterChi2);
        add(filterChi2);

        // row 15
        add(btnFilter);
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));

        // row 16
        add(btnReset);
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
        add(createJLabel(" ", ""));
    }

    /**
     * Inner class representing the filter text fields (min and max) for each parameter.
     */
    private static class FilterTextField {
        private final FitModel.Threshold threshold;
        private JTextField min;
        private JTextField max;

        /**
         * Constructs a FilterTextField with the specified threshold.
         *
         * @param threshold the threshold associated with this text field
         */
        public FilterTextField(FitModel.Threshold threshold) {
            this.threshold = threshold;

            min = createTextField(threshold.getMin(), "", createFocusListener(threshold::setMin));
            max = createTextField(threshold.getMax(), "", createFocusListener(threshold::setMax));
            this.setEnabled(false);
        }

        public void reset() {
            setText(min, threshold.getMin());
            setText(max, threshold.getMax());
            setEnabled(false);
        }

        /**
         * Enables or disables the text fields based on the provided flag.
         *
         * @param enabled true to enable the text fields, false to disable
         */
        public void setEnabled(boolean enabled) {
            min.setEnabled(enabled);
            max.setEnabled(enabled);
        }
    }
}
