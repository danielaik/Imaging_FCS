package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.DiffusionLawController;
import fiji.plugin.imaging_fcs.new_imfcs.model.DiffusionLawModel;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.FieldListenerFactory.createFocusListener;
import static fiji.plugin.imaging_fcs.new_imfcs.view.ButtonFactory.createJButton;
import static fiji.plugin.imaging_fcs.new_imfcs.view.ButtonFactory.createJToggleButton;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.createTextField;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.setText;
import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

/**
 * The {@code DiffusionLawView} class represents the view component in the MVC pattern
 * for the diffusion law analysis feature. It handles the user interface, allowing users
 * to interact with the diffusion law model through various input fields and buttons.
 */
public class DiffusionLawView extends BaseView {
    private static final GridLayout DIFFUSION_LAW_LAYOUT = new GridLayout(8, 5);
    private static final Point DIFFUSION_LAW_POSITION =
            new Point(Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 300, 125);
    private static final Dimension DIFFUSION_LAW_DIMENSION = new Dimension(350, 150);

    private final DiffusionLawModel model;
    private final DiffusionLawController controller;

    private JTextField tfDLBinStart, tfDLBinEnd, tfDLFitStart, tfDLFitEnd, tfDLROI;
    private JButton btnDFCalculate, btnDFFit;
    private JToggleButton tbDLRoi;

    /**
     * Constructs a new {@code DiffusionLawView} object.
     *
     * @param controller the controller responsible for handling user interactions
     * @param model      the model that contains the data for the diffusion law analysis
     */
    public DiffusionLawView(DiffusionLawController controller, DiffusionLawModel model) {
        super("Diffusion Law Analysis");

        this.model = model;
        this.controller = controller;

        initializeUI();
    }

    @Override
    protected void configureWindow() {
        super.configureWindow();

        setLayout(DIFFUSION_LAW_LAYOUT);
        setLocation(DIFFUSION_LAW_POSITION);
        setSize(DIFFUSION_LAW_DIMENSION);

        setVisible(false);
    }

    @Override
    protected void initializeTextFields() {
        tfDLBinStart = createTextField(model.getBinningStart(), "", createFocusListener(model::setBinningStart));
        tfDLBinEnd = createTextField(model.getBinningEnd(), "", createFocusListener(model::setBinningEnd));
        tfDLFitStart = createTextField(model.getFitStart(), "", createFocusListener(model::setFitStart));
        tfDLFitEnd = createTextField(model.getFitEnd(), "", createFocusListener(model::setFitEnd));

        tfDLROI = createTextField(model.getDimensionRoi(), "");
        tfDLROI.setEditable(false);
    }

    /**
     * Sets the editability of certain fields in the view based on the user's selection.
     *
     * @param b {@code true} to make the fields editable; {@code false} to make them non-editable.
     */
    public void setFieldsEditable(boolean b) {
        tfDLBinEnd.setEditable(b);
        tfDLFitStart.setEditable(b);
        tfDLFitEnd.setEditable(b);

        setText(tfDLBinEnd, model.getBinningEnd());
        setText(tfDLFitStart, model.getFitStart());
        setText(tfDLFitEnd, model.getFitEnd());
    }

    /**
     * Refresh the text fields for the binning and fit ranges.
     */
    public void setDefaultRange() {
        setText(tfDLBinStart, model.getBinningStart());
        setText(tfDLBinEnd, model.getBinningEnd());
        setText(tfDLFitStart, model.getFitStart());
        setText(tfDLFitEnd, model.getFitEnd());
    }


    @Override
    protected void initializeButtons() {
        btnDFCalculate = createJButton("Cal", "Calculate the diffusion law between the given 'Start-End' range.", null,
                controller.btnCalculatePressed());
        btnDFFit = createJButton("Fit", "Fit the diffusion law between the given 'Start-End' range.", null,
                controller.btnFitPressed());
        tbDLRoi = createJToggleButton(model.getMode(),
                "Set whether Diffusion Law is to be calculated over the full image or over smaller ROIs.", null,
                controller.tbDLRoiPressed());
    }

    @Override
    protected void addComponentsToFrame() {
        // row 1
        add(createJLabel("Calc", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));

        // row 2
        add(createJLabel("Binning:", ""));
        add(createJLabel("Start", ""));
        add(tfDLBinStart);
        add(createJLabel("End", ""));
        add(tfDLBinEnd);

        // row 3
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(btnDFCalculate);

        // row 4
        add(createJLabel("Fitting", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));

        // row 5
        add(createJLabel("Binning:", ""));
        add(createJLabel("Start", ""));
        add(tfDLFitStart);
        add(createJLabel("End", ""));
        add(tfDLFitEnd);

        // row 6
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(btnDFFit);

        // row 7
        add(createJLabel("ROI", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));

        // row 8
        add(tbDLRoi);
        add(createJLabel("", ""));
        add(tfDLROI);
        add(createJLabel("", ""));
        add(createJLabel("", ""));
    }
}
