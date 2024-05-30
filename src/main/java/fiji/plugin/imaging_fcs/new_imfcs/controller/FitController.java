package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.FitView;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class FitController {
    private final FitModel model;
    private final FitView view;

    public FitController(FitModel model) {
        this.model = model;
        this.view = new FitView(this, model);
    }

    public void setVisible(boolean b) {
        view.setVisible(b);
    }

    public ActionListener btnResetParametersPressed() {
        return (ActionEvent ev) -> {
            model.setDefaultValues();
            view.setDefaultValues();
        };
    }
}
