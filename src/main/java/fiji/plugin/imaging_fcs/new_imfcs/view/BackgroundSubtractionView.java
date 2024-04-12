package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.BackgroundSubtractionController;
import ij.IJ;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

/**
 * Sets up a dialog for selecting a background subtraction method and configuring its parameters.
 * Users can choose among several methods, input constant background values, and see the status of background
 * image loading. There's also an option to specify if subtraction should be applied after bleach correction.
 */
public final class BackgroundSubtractionView extends BaseView {
    private static final GridLayout BACKGROUND_SUBTRACTION_LAYOUT = new GridLayout(4, 2);
    private static final Point BACKGROUND_SUBTRACTION_LOCATION = new Point(800, 355);
    private static final Dimension BACKGROUND_SUBTRACTION_DIM = new Dimension(450, 180);

    private final BackgroundSubtractionController controller;

    // UI components
    private JComboBox<String> cbBackgroundSubtractionMethod;
    private JTextField tfBackground, tfBackground2, tfBGRLoadStatus;
    private JRadioButton rbtnIsSubtractionAfterBleachCorrection;

    /**
     * Initializes the view with the specified background subtraction controller.
     *
     * @param controller Controls background subtraction operations.
     */
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

    /**
     * Updates the status text and color based on the success or failure of background image loading.
     *
     * @param loaded Indicates whether the background image was successfully loaded.
     */
    public void updateStatusOnImageLoad(boolean loaded) {
        if (loaded) {
            tfBGRLoadStatus.setText("Successfully loaded background file.");
            tfBGRLoadStatus.setForeground(Color.BLUE);
        } else {
            tfBGRLoadStatus.setText("Fail to load background file.");
            tfBGRLoadStatus.setForeground(Color.RED);
        }
    }

    /**
     * Enables or disables the text fields for entering background values.
     *
     * @param enabled If true, the fields are enabled; otherwise, they are disabled.
     */
    public void setEnableBackgroundTextField(boolean enabled) {
        tfBackground.setEnabled(enabled);
        tfBackground2.setEnabled(enabled);
    }

    /**
     * Unselects the radio button for subtraction after bleach correction with a warning message.
     */
    public void unselectSubtractionAfterBleachCorrection() {
        if (rbtnIsSubtractionAfterBleachCorrection.isSelected()) {
            rbtnIsSubtractionAfterBleachCorrection.setSelected(false);
            IJ.showMessage("Unable to perform background subtraction on bleach corrected intensity traces.");
        }
    }
}
