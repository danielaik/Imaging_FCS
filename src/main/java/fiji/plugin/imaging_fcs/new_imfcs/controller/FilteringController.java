package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.OptionsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.SelectedPixel;
import fiji.plugin.imaging_fcs.new_imfcs.view.FilteringView;
import fiji.plugin.imaging_fcs.new_imfcs.view.Plots;
import ij.IJ;
import ij.ImagePlus;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;

/**
 * Controls filtering operations, connecting the view, models, and other controllers
 * for fluorescence correlation spectroscopy (FCS) processing within the Fiji plugin.
 */
public class FilteringController {
    private final FilteringView view;
    private final ExpSettingsModel settings;
    private final OptionsModel optionsModel;
    private final ImageController imageController;
    private final FitController fitController;
    private final Correlator correlator;

    /**
     * Initializes the FilteringController with the provided models and controllers.
     *
     * @param settings        Experiment settings model.
     * @param optionsModel    Filtering options model.
     * @param imageController Image-related operations controller.
     * @param fitController   Data fitting controller.
     * @param fitModel        Data fitting model.
     * @param correlator      Correlation data processor.
     */
    public FilteringController(ExpSettingsModel settings, OptionsModel optionsModel, ImageController imageController,
                               FitController fitController, FitModel fitModel, Correlator correlator) {
        this.settings = settings;
        this.optionsModel = optionsModel;
        this.imageController = imageController;
        this.fitController = fitController;
        this.correlator = correlator;

        this.view = new FilteringView(this, fitModel);
    }

    /**
     * Filters pixel data and updates the plots based on the current settings and fit criteria.
     */
    private void filterAndPlot() {
        Plots.updateParameterMaps(correlator.getPixelModels(),
                settings.getConvertedImageDimension(imageController.getImageDimension()),
                settings::convertPointToBinning, imageController.imageParamClicked(), fitController,
                optionsModel.isPlotParaHist());

        List<PixelModel> pixelModelList =
                SelectedPixel.getPixelModelsInRoi(null, settings.getPixelBinning(), settings.getMinCursorPosition(),
                        settings::convertPointToBinning, correlator.getPixelModels(), fitController);

        imageController.plotMultiplePixelsModels(pixelModelList);
    }

    /**
     * Creates an ActionListener for the filtering button, updating parameter maps
     * or showing an error if unavailable.
     *
     * @return ActionListener for filtering button.
     */
    public ActionListener btnFilteringPressed() {
        return (ActionEvent) -> {
            if (Plots.imgParam != null && Plots.imgParam.isVisible()) {
                filterAndPlot();
            } else {
                IJ.showMessage("No parameter map available. Perform a correlation or load an experiment.");
            }
        };
    }

    /**
     * Creates an ActionListener for the reset button, resetting filters and clearing fields.
     *
     * @return ActionListener for reset button.
     */
    public ActionListener btnResetPressed() {
        return (ActionEvent) -> {
            fitController.resetFilters();
            view.resetFields();

            if (Plots.imgParam != null && Plots.imgParam.isVisible()) {
                filterAndPlot();
            }
        };
    }

    /**
     * Creates an ActionListener for loading a binary filter image, updating the button label upon success.
     *
     * @return An ActionListener for the load binary filter button.
     */
    public ActionListener btnLoadBinaryFilterPressed() {
        return (ActionEvent ev) -> {
            if (Plots.imgParam != null && Plots.imgParam.isVisible()) {
                JButton button = (JButton) ev.getSource();
                ImagePlus filterImage = fitController.loadBinaryFilteringImage();
                if (filterImage != null) {
                    button.setText("Loaded");

                    filterAndPlot();
                } else {
                    button.setText("Binary");
                }
            } else {
                IJ.showMessage("No parameter map available. Perform a correlation or load an experiment.");
            }
        };
    }

    /**
     * Sets the visibility of the filtering view.
     *
     * @param b True to make the view visible, false to hide.
     */
    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    /**
     * Disposes of the filtering view.
     */
    public void dispose() {
        view.dispose();
    }

    /**
     * Brings the filtering view to the front.
     */
    public void toFront() {
        view.toFront();
    }
}
