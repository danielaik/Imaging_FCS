package fiji.plugin.imaging_fcs.directCameraReadout.updater;

import fiji.plugin.imaging_fcs.directCameraReadout.gui.DirectCapturePanel;
import static fiji.plugin.imaging_fcs.directCameraReadout.gui.DirectCapturePanel.*;

public class Updater {

    public static void UpdateDimTextField() {
        tfoWidth.setText(Integer.toString(DirectCapturePanel.Common.oWidth));
        tfoHeight.setText(Integer.toString(DirectCapturePanel.Common.oHeight));
        tfoLeft.setText(Integer.toString(DirectCapturePanel.Common.oLeft));
        tfoTop.setText(Integer.toString(DirectCapturePanel.Common.oTop));
        tfoRight.setText(Integer.toString(DirectCapturePanel.Common.oRight));
        tfoBottom.setText(Integer.toString(DirectCapturePanel.Common.oBottom));
        tfPixelDimension.setText(Integer.toString(DirectCapturePanel.Common.oWidth) + " x " + Integer.toString(DirectCapturePanel.Common.oHeight));
    }

}
