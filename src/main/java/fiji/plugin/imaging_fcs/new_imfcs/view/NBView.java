package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.NBController;
import fiji.plugin.imaging_fcs.new_imfcs.model.NBModel;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.FocusListenerFactory.createFocusListener;
import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

public final class NBView extends BaseView {
    private static final GridLayout NB_LAYOUT = new GridLayout(4, 2);
    private static final Point NB_LOCATION = new Point(
            Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 300, 125);
    private static final Dimension NB_DIM = new Dimension(250, 150);

    private final NBController controller;
    private final NBModel model;

    // UI elements
    private JTextField tfNBS, tfNBCalibRatio;
    private JComboBox<String> cbNBMode;
    private JButton btnNB;

    public NBView(NBController controller, NBModel model) {
        super("N&B");
        this.controller = controller;
        this.model = model;

        initializeUI();
    }

    @Override
    protected void configureWindow() {
        super.configureWindow();

        setLayout(NB_LAYOUT);
        setLocation(NB_LOCATION);
        setSize(NB_DIM);

        setVisible(false);
    }

    @Override
    protected void initializeComboBoxes() {
        cbNBMode = new JComboBox<>(new String[]{"G1", "Calibrated"});
        cbNBMode.addActionListener(controller.cbNBModeChanged());
    }

    @Override
    protected void initializeButtons() {
        btnNB = ButtonFactory.createJButton(
                "N&B", "perform N&B analysis", null, controller.btnNBPressed());
    }

    @Override
    protected void initializeTextFields() {
        tfNBS = TextFieldFactory.createTextField(model.getS_value(), "", createFocusListener(model::setS_value));
        tfNBCalibRatio = TextFieldFactory.createTextField(model.getCalibRatio(), "",
                createFocusListener(model::setCalibRatio));

        tfNBS.setEnabled(false);
        tfNBCalibRatio.setEnabled(false);
    }

    @Override
    protected void addComponentsToFrame() {
        add(createJLabel("NB mode", ""));
        add(cbNBMode);

        add(createJLabel("Calib Ratio", ""));
        add(tfNBCalibRatio);

        add(createJLabel("N&B analysis", ""));
        add(btnNB);

        add(createJLabel("S value", ""));
        add(tfNBS);
    }

    public void setEnabledTextFields(boolean enabled) {
        tfNBS.setEnabled(enabled);
        tfNBCalibRatio.setEnabled(enabled);
    }
}
