package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.DiffusionLawModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.ImageModel;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Pair;
import fiji.plugin.imaging_fcs.new_imfcs.view.DiffusionLawView;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

/**
 * The {@code DiffusionLawController} class serves as the controller in the Model-View-Controller (MVC) pattern
 * for managing the diffusion law analysis workflow. This class mediates the interactions between the
 * {@link DiffusionLawModel} (the model) and the {@link DiffusionLawView} (the view), handling user inputs
 * and updating the model and view based on these inputs.
 */
public class DiffusionLawController {
    private final DiffusionLawView view;
    private final DiffusionLawModel model;

    /**
     * Constructs a new {@code DiffusionLawController} instance, initializing the associated
     * model and view for managing diffusion law analysis.
     *
     * @param settings   the experimental settings model containing the parameters for the analysis.
     * @param imageModel the image model containing the data to be analyzed.
     * @param fitModel   the fitting model used to fit the correlation data.
     */
    public DiffusionLawController(ExpSettingsModel settings, ImageModel imageModel, FitModel fitModel) {
        this.model = new DiffusionLawModel(settings, imageModel, fitModel);
        this.view = new DiffusionLawView(this, model);
    }

    /**
     * Creates an {@link ItemListener} to handle state changes of the toggle button in the diffusion law view.
     * The button toggles between "ROI" (Region of Interest) mode and "All" mode. This listener updates the model
     * and view based on the selected mode.
     *
     * @return An {@link ItemListener} that updates the model's range values and view's editability based on
     * the button's state (selected or deselected).
     */
    public ItemListener tbDLRoiPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);

            String mode = model.resetRangeValues(selected);
            view.setFieldsEditable(!selected);

            button.setText(mode);
        };
    }

    /**
     * Creates an {@link ActionListener} to handle the event when the "Calculate" button is pressed in the view.
     * This listener triggers the fitting process, retrieves the diffusion law results from the model,
     * and plots the results using the {@link Plots#plotDiffLaw(double[][], double, double)} method.
     *
     * @return An {@link ActionListener} that initiates the diffusion law fitting process and plots the results.
     */
    public ActionListener btnCalculatePressed() {
        return (ActionEvent ev) -> {
            Pair<Double, Double> minMax = model.calculateDiffusionLaw();
            Plots.plotDiffLaw(model.getEffectiveArea(), model.getTime(), model.getStandardDeviation(), minMax.getLeft(),
                    minMax.getRight());
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
