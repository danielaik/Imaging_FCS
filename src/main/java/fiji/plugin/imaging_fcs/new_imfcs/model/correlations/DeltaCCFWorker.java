package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import fiji.plugin.imaging_fcs.new_imfcs.enums.DccfDirection;
import fiji.plugin.imaging_fcs.new_imfcs.model.BackgroundTaskWorker;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import ij.IJ;
import ij.ImagePlus;

import java.util.function.BiConsumer;

/**
 * The DCCFWorker class extends BackgroundTaskWorker to perform the DCCF computation in a separate thread,
 * ensuring that the UI remains responsive.
 */
public final class DeltaCCFWorker extends BackgroundTaskWorker<double[][], Void> {
    private final ExpSettingsModel settings;
    private final Correlator correlator;
    private final ImagePlus image;
    private final DccfDirection direction;
    private final BiConsumer<double[][], DccfDirection> runOnCompletion;

    /**
     * Constructs a new DCCFWorker with the specified parameters.
     *
     * @param settings        The experimental settings model.
     * @param correlator      The correlator used to compute the pixel correlations.
     * @param image           The image to be processed.
     * @param direction       The direction for DCCF computation.
     * @param runOnCompletion A BiConsumer to run upon completion, accepting the computed DCCF array and direction
     *                        name for plotting.
     */
    public DeltaCCFWorker(ExpSettingsModel settings, Correlator correlator, ImagePlus image, DccfDirection direction,
                          BiConsumer<double[][], DccfDirection> runOnCompletion) {
        super(() -> {});

        this.settings = settings;
        this.correlator = correlator;
        this.image = image;
        this.direction = direction;
        this.runOnCompletion = runOnCompletion;
    }

    /**
     * Performs the DCCF computation in the background.
     *
     * @return A 2D array representing the computed DCCF values.
     * @throws Exception If an error occurs during computation.
     */
    @Override
    protected double[][] doInBackground() throws Exception {
        return DeltaCCF.dccf(correlator, image, direction, settings);
    }

    /**
     * Runs when the background computation is done, passing the result to the specified BiConsumer.
     */
    @Override
    protected void done() {
        try {
            runOnCompletion.accept(get(), direction);
        } catch (Exception e) {
            IJ.showMessage("Error", "DCCF crashed: " + e.getMessage());
        }
    }
}
