package fiji.plugin.imaging_fcs.imfcs.model;

import fiji.plugin.imaging_fcs.imfcs.controller.ImageController;
import ij.IJ;
import ij.gui.Roi;

import javax.swing.*;

/**
 * CorrelationWorker performs correlation on a given ROI using an ImageController.
 * It ensures that only one correlation task is active at a time by canceling any
 * previously running instance. The worker passes its cancellation status to the
 * correlation process so that it can exit early if needed.
 */
public class CorrelationWorker extends SwingWorker<Void, Void> {
    // Track the current running instance.
    private static volatile CorrelationWorker currentInstance;

    // References to the objects needed for correlation.
    private final ImageController imageController;
    private final Roi imgRoi;

    /**
     * Creates a new CorrelationWorker and cancels any existing instance.
     *
     * @param imageController the controller performing correlation
     * @param imgRoi          the region of interest to process
     */
    public CorrelationWorker(ImageController imageController, Roi imgRoi) {
        cancelPreviousInstance();

        this.imageController = imageController;
        this.imgRoi = imgRoi;
        currentInstance = this;
    }

    /**
     * Cancels the previous instance if it's still running
     */
    public static void cancelPreviousInstance() {
        synchronized (CorrelationWorker.class) {
            if (currentInstance != null && !currentInstance.isDone()) {
                currentInstance.cancel(true); // Interrupt the running task
            }
        }
    }

    /**
     * Performs the correlation in a background thread.
     * The ImageController's correlateROI method is called with a cancellation
     * checker that returns this worker's cancellation status.
     *
     * @return null upon completion
     * @throws Exception if an error occurs during correlation
     */
    @Override
    protected Void doInBackground() throws Exception {
        imageController.correlateROI(imgRoi, this::isCancelled);
        return null;
    }

    /**
     * Updates the UI when the task is finished.
     * If the task was cancelled, the status is updated accordingly.
     * The static reference to the current instance is cleared.
     */
    @Override
    protected void done() {
        try {
            // Handle cancellation vs normal completion
            if (isCancelled()) {
                SwingUtilities.invokeLater(() -> {
                    IJ.showStatus("Task cancelled");
                    IJ.showProgress(0);
                });
            } else {
                super.done(); // Call parent's done() for normal completion
            }
        } finally {
            synchronized (CorrelationWorker.class) {
                if (currentInstance == this) {
                    currentInstance = null; // Clear reference when done
                }
            }
        }
    }
}