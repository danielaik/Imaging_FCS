package fiji.plugin.imaging_fcs.imfcs.view.dialogs;

import fiji.plugin.imaging_fcs.imfcs.model.ParameterVideoModel;
import ij.gui.GenericDialog;

import java.util.function.Consumer;

/**
 * A dialog for configuring video parameters, including frame range, step size,
 * and save options.
 */
public class ParameterVideoView extends GenericDialog {
    /**
     * Initializes the dialog with the given frame range and model, and
     * passes the view to the listener if confirmed.
     *
     * @param firstFrame the first frame of the video
     * @param lastFrame  the last frame of the video
     * @param model      the video parameter model
     * @param listener   consumer to handle confirmed input
     */
    public ParameterVideoView(int firstFrame, int lastFrame, ParameterVideoModel model,
                              Consumer<ParameterVideoView> listener) {
        super("ParamVideo");
        displayDialog(firstFrame, lastFrame, model, listener);
    }

    /**
     * Displays the dialog for setting video parameters and invokes the listener on confirmation.
     *
     * @param firstFrame the first frame of the video
     * @param lastFrame  the last frame of the video
     * @param model      the video parameter model
     * @param listener   consumer to handle confirmed input
     */
    private void displayDialog(int firstFrame, int lastFrame, ParameterVideoModel model,
                               Consumer<ParameterVideoView> listener) {
        int length = (lastFrame - firstFrame + 1) / 10;
        int step = 10000;

        addNumericField("Start frame: ", firstFrame, 0);
        addNumericField("End frame: ", lastFrame, 0);
        addNumericField("Length: ", length, 0);
        addNumericField("Step size: ", step, 0);

        addCheckbox("Save CF and fits: ", model.isSaveCFAndFitPVideo());
        addStringField("Video name: ", "");

        showDialog();

        if (wasCanceled()) {
            return;
        }

        if (wasOKed()) {
            listener.accept(this);
        }
    }
}
