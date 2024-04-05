package fiji.plugin.imaging_fcs.new_imfcs.view;

import javax.swing.*;

public abstract class BaseView extends JFrame {
    public BaseView(String title) {
        super(title);
    }

    protected final void initializeUI() {
        configureWindow();

        initializeComboBoxes();
        initializeTextFields();
        initializeButtons();

        addComponentsToFrame();
    }

    protected void configureWindow() {
        setFocusable(true);
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        setResizable(false);
    }

    protected void initializeComboBoxes() {
    }

    protected void initializeTextFields() {
    }

    protected void initializeButtons() {
    }

    protected abstract void addComponentsToFrame();
}
