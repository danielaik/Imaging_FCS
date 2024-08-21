package fiji.plugin.imaging_fcs.new_imfcs.model;

import ij.IJ;

import javax.swing.*;

/**
 * The BackgroundTaskWorker class extends SwingWorker to perform a background task.
 * This class takes a Runnable that defines the task logic and executes it in a background thread,
 * updating the status upon completion.
 */
public class BackgroundTaskWorker extends SwingWorker<Void, Void> {
    private final Runnable task;

    /**
     * Constructs a BackgroundTaskWorker with the specified task.
     *
     * @param task the Runnable that defines the task logic
     */
    public BackgroundTaskWorker(Runnable task) {
        this.task = task;
    }

    /**
     * Performs the background computation.
     * This method is executed in a background thread.
     *
     * @return null upon completion
     */
    @Override
    protected Void doInBackground() throws Exception {
        task.run();
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
