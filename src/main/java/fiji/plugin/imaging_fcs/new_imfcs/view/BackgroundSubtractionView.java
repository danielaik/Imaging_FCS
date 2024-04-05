package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.BackgroundSubtractionController;
import ij.IJ;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

public class BackgroundSubtractionView extends BaseView {
    private static final GridLayout BACKGROUND_SUBTRACTION_LAYOUT = new GridLayout(4, 2);
    private static final Point BACKGROUND_SUBTRACTION_LOCATION = new Point(800, 355);
    private static final Dimension BACKGROUND_SUBTRACTION_DIM = new Dimension(450, 180);

    private final BackgroundSubtractionController controller;

    private JComboBox<String> cbBackgroundSubtractionMethod;

    private JTextField tfBackground;
    private JTextField tfBackground2;
    private JTextField tfBGRLoadStatus;

    private JRadioButton rbtnIsSubtractionAfterBleachCorrection;

    public BackgroundSubtractionView(BackgroundSubtractionController controller) {
        super("Background subtraction method selection");
        this.controller = controller;

        initializeUI();
    }

    @Override
    protected void configureWindow() {
        super.configureWindow();

        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        setLayout(BACKGROUND_SUBTRACTION_LAYOUT);
        setLocation(BACKGROUND_SUBTRACTION_LOCATION);
        setSize(BACKGROUND_SUBTRACTION_DIM);

        setVisible(false);
    }

    @Override
    protected void initializeComboBoxes() {
        cbBackgroundSubtractionMethod = new JComboBox<>(new String[]{
                "Constant Background", "Min frame by frame", "Min per image stack", "Min Pixel wise per image stack",
                "Load BGR image"
        });
        cbBackgroundSubtractionMethod.addActionListener(controller.cbBackgroundSubtractionMethodChanged());
    }

    @Override
    protected void initializeTextFields() {
        tfBackground = TextFieldFactory.createTextField("0", "");
        tfBackground2 = TextFieldFactory.createTextField("0", "");

        tfBGRLoadStatus = TextFieldFactory.createTextField("",
                "Status on background file for correction. Has to be the same area recorded under the same conditions as the experimental file");
        tfBGRLoadStatus.setEditable(false);
        tfBGRLoadStatus.setFont(new Font(Constants.PANEL_FONT, Font.BOLD, Constants.PANEL_FONT_SIZE));
    }

    @Override
    protected void initializeButtons() {
        rbtnIsSubtractionAfterBleachCorrection = new JRadioButton("Subtract after bleach correction");
    }

    @Override
    protected void addComponentsToFrame() {
        add(createJLabel("Background correction method: ", ""));
        add(cbBackgroundSubtractionMethod);

        add(createJLabel("Background 1: ", ""));
        add(tfBackground);

        add(createJLabel("Background 2: ", ""));
        add(tfBackground2);

        add(rbtnIsSubtractionAfterBleachCorrection);
        add(tfBGRLoadStatus);
    }

    public void updateStatusOnImageLoad(boolean loaded) {
        if (loaded) {
            tfBGRLoadStatus.setText("Successfully loaded background file.");
            tfBGRLoadStatus.setForeground(Color.BLUE);
        } else {
            tfBGRLoadStatus.setText("Fail to load background file.");
            tfBGRLoadStatus.setForeground(Color.RED);
        }
    }

    public void setEnableBackgroundTextField(boolean enabled) {
        tfBackground.setEnabled(enabled);
        tfBackground2.setEnabled(enabled);
    }

    public void unselectSubtractionAfterBleachCorrection() {
        if (rbtnIsSubtractionAfterBleachCorrection.isSelected()) {
            rbtnIsSubtractionAfterBleachCorrection.setSelected(false);
            IJ.showMessage("Unable to perform background subtraction on bleach corrected intensity traces.");
        }
    }
}
