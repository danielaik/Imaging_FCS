package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.FitView;

public class FitController {
    private final FitModel model;
    private final FitView view;

    public FitController(FitModel model) {
        this.model = model;
        this.view = new FitView(model);
    }

    public void setVisible(boolean b) {
        view.setVisible(b);
    }
}
