package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.DiffusionLawModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.DiffusionLawView;

import javax.swing.*;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

/**
 * The {@code DiffusionLawController} class acts as a controller in the MVC pattern,
 * coordinating interactions between the {@link DiffusionLawView} and {@link DiffusionLawModel}.
 * It manages user inputs and updates the model and view accordingly.
 */
public class DiffusionLawController {
    private final DiffusionLawView view;
    private final DiffusionLawModel model;

    /**
     * Constructs a new {@code DiffusionLawController} object.
     * Initializes the associated model and view for the diffusion law functionality.
     */
    public DiffusionLawController() {
        this.model = new DiffusionLawModel();
        this.view = new DiffusionLawView(this, model);
    }

    /**
     * Creates an {@link ItemListener} for handling the state changes of a toggle button
     * in the diffusion law view. When the button is pressed, it toggles between "ROI" (Region of Interest)
     * and "All", updating the model and view accordingly.
     *
     * @return An {@link ItemListener} that updates the button text and view state based on the selected
     * range (ROI or All).
     */
    public ItemListener tbDLRoiPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setText(selected ? "ROI" : "All");

            if (selected) {
                model.resetRangeValues();
            }

            view.setFieldsEditable(!selected);
        };
    }

    /**
     * Sets the visibility of the diffusion law view.
     *
     * @param b {@code true} to make the view visible; {@code false} to hide it.
     */
    public void setVisible(boolean b) {
        view.setVisible(b);
    }
}
