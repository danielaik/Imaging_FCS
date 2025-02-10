package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.*;
import fiji.plugin.imaging_fcs.new_imfcs.utils.ExcelExporter;
import fiji.plugin.imaging_fcs.new_imfcs.utils.Pair;
import fiji.plugin.imaging_fcs.new_imfcs.view.DiffusionLawView;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.PSFView;
import ij.IJ;
import org.apache.poi.ss.usermodel.Workbook;

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
     * @param settings              the experimental settings model containing the parameters for the analysis.
     * @param imageModel            the image model containing the data to be analyzed.
     * @param fitModel              the fitting model used to fit the correlation data.
     * @param bleachCorrectionModel the bleach correction model used to apply bleach correction.
     */
    public DiffusionLawController(ExpSettingsModel settings, ImageModel imageModel, FitModel fitModel,
                                  BleachCorrectionModel bleachCorrectionModel) {
        this.model =
                new DiffusionLawModel(settings, imageModel, fitModel, bleachCorrectionModel, this::askResetResults);
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
     * Executes the diffusion law calculation and updates the plot with the results.
     * Calls the model to perform the calculation and then plots the effective area against time.
     */
    public void runCalculate() {
        model.calculateDiffusionLaw();
        Plots.plotDiffLaw(model.getEffectiveArea(), model.getTime(), model.getStandardDeviation(),
                model.getMinValueDiffusionLaw(), model.getMaxValueDiffusionLaw());
    }

    /**
     * Creates a listener for the "Calculate" button, triggering the diffusion law calculation and plotting the results.
     *
     * @return Listener that initiates the calculation and updates the plot.
     */
    public ActionListener btnCalculatePressed() {
        return (ActionEvent ev) -> new BackgroundTaskWorker<Void, Void>(() -> {
            IJ.showStatus("Calculating diffusion law");
            runCalculate();
        }).execute();
    }

    /**
     * Creates a listener for the "Fit" button, performing the fit and updating the plot.
     *
     * @return Listener that fits the data and updates the plot.
     */
    public ActionListener btnFitPressed() {
        return (ActionEvent ev) -> {
            if (!Plots.isPlotDiffLawOpen()) {
                IJ.showMessage("No window open, please run the calculation before");
            } else {
                try {
                    double[][] fitFunction = model.fit();
                    Plots.plotDiffLaw(model.getEffectiveArea(), model.getTime(), model.getStandardDeviation(),
                            model.getMinValueDiffusionLaw(), model.getMaxValueDiffusionLaw());
                    Plots.plotFitDiffLaw(model.getIntercept(), model.getSlope(), fitFunction);
                } catch (RuntimeException e) {
                    IJ.showMessage(e.getMessage());
                }
            }
        };
    }

    /**
     * Prompts the user to confirm whether they want to reset the results due to a change in the binning range.
     * If confirmed, the model's results are reset, and the diffusion law plot window is closed.
     * If the user rejects the reset, an exception is thrown to cancel the operation.
     */
    private void askResetResults() {
        if (Plots.isPlotDiffLawOpen()) {
            int response = JOptionPane.showConfirmDialog(null,
                    "The binning range has changed since your last calculations.\n" +
                            "Continuing will result in deleting these results.", "Delete the Results and start new?",
                    JOptionPane.YES_NO_OPTION);

            if (response == JOptionPane.YES_OPTION) {
                model.resetResults();
                Plots.closeDiffLawWindow();
            } else {
                throw new RejectResetException();
            }
        }
    }

    /**
     * Set the default ranges for binning and fit based on the image model dimension.
     */
    public void setDefaultRange() {
        model.setDefaultRange();
        view.setDefaultRange();
    }

    /**
     * Saves the diffusion law and PSF data into the given Excel workbook.
     * This method creates and adds sheets for diffusion law data and PSF results using the corresponding model.
     *
     * @param workbook the Excel workbook to which the sheets will be added
     */
    public void saveExcelSheets(Workbook workbook) {
        ExcelExporter.saveDiffusionLawSheet(workbook, model);
        ExcelExporter.savePSFSheet(workbook, model.getPsfResults());
    }

    /**
     * Sets the visibility of the diffusion law view.
     *
     * @param b {@code true} to make the view visible; {@code false} to hide it.
     */
    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    /**
     * Displays the Point Spread Function (PSF) calculation dialog to the user.
     */
    public void displayPSFDialog() {
        new PSFView(this::runPSF);
    }

    /**
     * Runs the PSF calculation using the parameters provided by the PSF dialog.
     *
     * @param view The PSF dialog view containing user inputs.
     */
    private void runPSF(PSFView view) {
        double startValue = view.getNextNumber();
        double endValue = view.getNextNumber();
        double stepSize = view.getNextNumber();

        int binningStart = (int) view.getNextNumber();
        int binningEnd = (int) view.getNextNumber();

        new BackgroundTaskWorker<Void, Void>(() -> {
            IJ.showStatus("Calculating PSF");
            try {
                Pair<Double, Double> minMax =
                        model.determinePSF(startValue, endValue, stepSize, binningStart, binningEnd);
                Plots.plotPSF(minMax.getLeft(), minMax.getRight(), model.getPsfResults(), binningStart, binningEnd);
            } catch (Exception e) {
                IJ.showMessage(e.getMessage());
            }
        }).execute();
    }

    /**
     * Dispose the view
     */
    public void dispose() {
        this.view.dispose();
    }
}
