package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.NBController;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

public final class NBView extends BaseView {
    private static final GridLayout NB_LAYOUT = new GridLayout(4, 2);
    private static final Point NB_LOCATION = new Point(
            Constants.MAIN_PANEL_POS.x + Constants.MAIN_PANEL_DIM.width + 300, 125);
    private static final Dimension NB_DIM = new Dimension(250, 150);

    private final NBController controller;

    // UI elements
    private JTextField tfNBS, tfNBCalibRatio;
    private JComboBox<String> cbNBMode;
    private JButton btnNB;

    public NBView(NBController controller) {
        super("N&B");
        this.controller = controller;

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
        tfNBS = TextFieldFactory.createTextField("0.0", "");
        tfNBCalibRatio = TextFieldFactory.createTextField("2", "");

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
