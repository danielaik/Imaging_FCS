package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;
import fiji.plugin.imaging_fcs.new_imfcs.view.FitView;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.function.Consumer;

public class FitController {
    private final FitModel model;
    private final FitView view;

    public FitController(FitModel model) {
        this.model = model;
        this.view = new FitView(this, model);
    }

    public void setVisible(boolean b) {
        model.setActivated(b);
        view.setVisible(b);
    }

    public ActionListener btnResetParametersPressed() {
        return (ActionEvent ev) -> {
            model.setDefaultValues();
            view.setDefaultValues();
        };
    }

    public ItemListener tbFixParPressed() {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setText(selected ? "Fix" : "Free");
            model.setFix(selected);
        };
    }

    public ItemListener tbOptionPressed(Consumer<Boolean> setter) {
        return (ItemEvent ev) -> {
            JToggleButton button = (JToggleButton) ev.getItemSelectable();

            boolean selected = (ev.getStateChange() == ItemEvent.SELECTED);
            button.setForeground(selected ? Color.BLACK : Color.LIGHT_GRAY);
            setter.accept(selected);
        };
    }
}
