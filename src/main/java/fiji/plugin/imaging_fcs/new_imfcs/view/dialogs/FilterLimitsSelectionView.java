package fiji.plugin.imaging_fcs.new_imfcs.view.dialogs;

import ij.gui.GenericDialog;

import java.util.function.Consumer;

/**
 * This class provides a user interface for selecting filter limits.
 */
public final class FilterLimitsSelectionView extends GenericDialog {
    private final Consumer<FilterLimitsSelectionView> listener;
    private final int filterLowerLimit;
    private final int filterUpperLimit;

    /**
     * Creates a dialog for filter limit selection.
     *
     * @param listener         the callback to handle the selected limits
     * @param filterLowerLimit the initial value for the lower limit field
     * @param filterUpperLimit the initial value for the upper limit field
     */
    public FilterLimitsSelectionView(Consumer<FilterLimitsSelectionView> listener, int filterLowerLimit,
                                     int filterUpperLimit) {
        super("Filter");
        this.listener = listener;
        this.filterLowerLimit = filterLowerLimit;
        this.filterUpperLimit = filterUpperLimit;

        displayDialog();
    }

    /**
     * Sets up and displays the dialog.
     * It requires the user to input non-negative integer values for upper and lower limits.
     * Calls the listener if the user confirms their choices.
     */
    private void displayDialog() {
        addMessage("Non-negative integers only");
        addNumericField("Lower limit: ", filterLowerLimit, 0);
        addNumericField("Upper limit: ", filterUpperLimit, 0);

        hideCancelButton();
        showDialog();

        if (wasOKed()) {
            listener.accept(this);
        }
    }
}
