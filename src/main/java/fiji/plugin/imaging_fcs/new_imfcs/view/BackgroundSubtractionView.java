package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.BackgroundSubtractionController;
import fiji.plugin.imaging_fcs.new_imfcs.enums.BackgroundMode;
import fiji.plugin.imaging_fcs.new_imfcs.model.BackgroundModel;

import javax.swing.*;
import java.awt.*;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.FieldListenerFactory.createFocusListener;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.createTextField;
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
    private final BackgroundModel model;

    // UI components
    private JComboBox<BackgroundMode> cbBackgroundSubtractionMethod;
    private JTextField tfBackground, tfBackground2, tfBGRLoadStatus;

    /**
     * Initializes the view with the specified background subtraction controller.
     *
     * @param controller Controls background subtraction operations.
     */
    public BackgroundSubtractionView(BackgroundSubtractionController controller, BackgroundModel model) {
        super("Background subtraction method selection");
        this.controller = controller;
        this.model = model;

        initializeUI();
    }

    @Override
    protected void configureWindow() {
        super.configureWindow();

        setLayout(BACKGROUND_SUBTRACTION_LAYOUT);
        setLocation(BACKGROUND_SUBTRACTION_LOCATION);
        setSize(BACKGROUND_SUBTRACTION_DIM);

        setVisible(false);
    }

    @Override
    protected void initializeComboBoxes() {
        cbBackgroundSubtractionMethod = new JComboBox<>(BackgroundMode.values());
        cbBackgroundSubtractionMethod.setSelectedItem(model.getMode());
        cbBackgroundSubtractionMethod.addActionListener(
                controller.cbBackgroundSubtractionMethodChanged(cbBackgroundSubtractionMethod));
    }

    @Override
    protected void initializeTextFields() {
        tfBackground =
                createTextField(model.getConstantBackground1(), "", createFocusListener(model::setConstantBackground1));
        tfBackground2 =
                createTextField(model.getConstantBackground2(), "", createFocusListener(model::setConstantBackground2));

        tfBGRLoadStatus = createTextField("",
                "Status on background file for correction. Has to be the same area recorded under the same conditions" +
                        " as the experimental file");
        tfBGRLoadStatus.setEditable(false);
        tfBGRLoadStatus.setFont(new Font(Constants.PANEL_FONT, Font.BOLD, Constants.PANEL_FONT_SIZE));
    }

    @Override
    protected void addComponentsToFrame() {
        add(createJLabel("Background correction method: ", ""));
        add(cbBackgroundSubtractionMethod);

        add(createJLabel("Background 1: ", ""));
        add(tfBackground);

        add(createJLabel("Background 2: ", ""));
        add(tfBackground2);

        add(createJLabel("", ""));
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

    public void setTfBackground(String text) {
        tfBackground.setText(text);
    }

    public void setTfBackground2(String text) {
        tfBackground2.setText(text);
    }
}
