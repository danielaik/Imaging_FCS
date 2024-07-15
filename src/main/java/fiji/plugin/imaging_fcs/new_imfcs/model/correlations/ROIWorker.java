package fiji.plugin.imaging_fcs.new_imfcs.model.correlations;

import ij.IJ;

import javax.swing.*;

/**
 * The ROIWorker class extends SwingWorker to perform a background task
 * that correlates a region of interest (ROI) in an image.
 * This class takes a Runnable that defines the correlation logic
 * and executes it in a background thread, updating the status upon completion.
 */
public class ROIWorker extends SwingWorker<Void, Void> {
    private final Runnable correlateROIFunction;

    /**
     * Constructs an ROIWorker with the specified correlation function.
     *
     * @param correlateROIFunction the Runnable that defines the correlation logic
     */
    public ROIWorker(Runnable correlateROIFunction) {
        this.correlateROIFunction = correlateROIFunction;
    }

    /**
     * Performs the background computation.
     * This method is executed in a background thread.
     *
     * @return null upon completion
     */
    @Override
    protected Void doInBackground() throws Exception {
        correlateROIFunction.run();
        return null;
    }

    /**
     * This method is called when the background computation is finished.
     * It updates the status to indicate completion.
     */
    @Override
    protected void done() {
        IJ.showStatus("Done");
    }
}
