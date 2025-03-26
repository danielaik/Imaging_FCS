package fiji.plugin.imaging_fcs.imfcs.view.dialogs;

import ij.gui.GenericDialog;

import java.util.function.Consumer;

/**
 * The {@code PSFView} class creates a dialog for configuring Point Spread Function (PSF) calculation parameters.
 * It extends the {@link GenericDialog} class from ImageJ, providing fields for the user to input the start and end
 * values,
 * step size, and binning range for the PSF calculation. When the dialog is confirmed, the input values are passed
 * to a specified listener for processing.
 */
public class PSFView extends GenericDialog {
    /**
     * Constructs a {@code PSFView} dialog with the given listener to process the user input.
     *
     * @param listener a {@link Consumer} that accepts this {@code PSFView} instance if the user confirms the dialog.
     */
    public PSFView(Consumer<PSFView> listener) {
        super("PSF");
        displayDialog(listener);
    }

    /**
     * Displays the PSF configuration dialog to the user.
     * This method adds numeric fields for the start value, end value, step size, and binning range.
     * If the dialog is confirmed (OKed), the provided listener is invoked with this {@code PSFView} instance.
     *
     * @param listener a {@link Consumer} that processes the input values if the dialog is confirmed.
     */
    private void displayDialog(Consumer<PSFView> listener) {
        addNumericField("Start value:", 0.6, 2);
        addNumericField("End value: ", 1.0, 2);
        addNumericField("Step size: ", 0.1, 2);
        addNumericField("Binning start value: ", 1, 0);
        addNumericField("Binning end value: ", 5, 0);

        showDialog();

        if (wasCanceled()) {
            return;
        }

        if (wasOKed()) {
            listener.accept(this);
        }
    }
}
