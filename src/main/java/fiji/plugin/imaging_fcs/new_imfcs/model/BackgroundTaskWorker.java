package fiji.plugin.imaging_fcs.new_imfcs.model;

import ij.IJ;

import javax.swing.*;
import java.util.concurrent.ExecutionException;

/**
 * The BackgroundTaskWorker class extends SwingWorker to perform a background task.
 * This class takes a Runnable that defines the task logic and executes it in a background thread,
 * updating the status upon completion.
 */
public class BackgroundTaskWorker<T, V> extends SwingWorker<T, V> {
    protected final Runnable task;

    /**
     * Constructs a BackgroundTaskWorker with the specified task.
     *
     * @param task the Runnable that defines the task logic
     */
    public BackgroundTaskWorker(Runnable task) {
        IJ.showProgress(0);
        this.task = task;
    }

    /**
     * Performs the background computation.
     * This method is executed in a background thread.
     *
     * @return null upon completion
     */
    @Override
    protected T doInBackground() throws Exception {
        task.run();
        return null;
    }

    /**
     * Executes the background task and waits for its completion.
     * Starts the task asynchronously and then blocks until the task finishes.
     * Any interruptions or execution exceptions are caught and logged.
     */
    public void executeAndWait() {
        this.execute();
        try {
            // wait for the task to complete
            this.get();
        } catch (InterruptedException | ExecutionException e) {
            IJ.log(e.getMessage());
        }
    }

    /**
     * This method is called when the background computation is finished.
     * It updates the status to indicate completion.
     */
    @Override
    protected void done() {
        IJ.showProgress(1);
        IJ.showStatus("Done");
    }
}
