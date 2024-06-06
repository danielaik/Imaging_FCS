package fiji.plugin.imaging_fcs.new_imfcs.view;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.controller.FitController;
import fiji.plugin.imaging_fcs.new_imfcs.model.FitModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ItemListener;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.FocusListenerFactory.createFocusListener;
import static fiji.plugin.imaging_fcs.new_imfcs.view.ButtonFactory.createJButton;
import static fiji.plugin.imaging_fcs.new_imfcs.view.ButtonFactory.createJToggleButton;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.createTextField;
import static fiji.plugin.imaging_fcs.new_imfcs.view.TextFieldFactory.setText;
import static fiji.plugin.imaging_fcs.new_imfcs.view.UIUtils.createJLabel;

public class FitView extends BaseView {
    private static final GridLayout FIT_LAYOUT = new GridLayout(17, 6);
    private static final Point FIT_LOCATION =
            new Point(10, Constants.MAIN_PANEL_POS.y + Constants.MAIN_PANEL_DIM.height + 10);
    private static final Dimension FIT_DIMENSION = new Dimension(370, 370);
    private final FitModel model;
    private final FitController controller;

    private JTextField tfParamQ2, tfParamN, tfParamF2, tfParamD, tfParamD2, tfParamF3, tfParamD3, tfParamQ3, tfParamVx,
            tfParamVy, tfParamG, tfParamFTrip, tfParamTTrip, tfFitStart, tfFitEnd, tfFitModel, tfModProb1, tfModProb2,
            tfModProb3;
    private JButton btnTest, btnSetPar, btnCNNImage, btnCNNACF;
    private JToggleButton tbGLS, tbBayes, tbFixPar;
    private JRadioButton holdN, holdF2, holdD, holdD2, holdF3, holdD3, holdVx, holdVy, holdG, holdFTrip, holdTTtrip,
            rbtnCNNImage, rbtnCNNACF;


    public FitView(FitController controller, FitModel model) {
        super("Fit");
        this.controller = controller;
        this.model = model;
        initializeUI();
    }

    @Override
    protected void configureWindow() {
        super.configureWindow();

        setLayout(FIT_LAYOUT);
        setLocation(FIT_LOCATION);
        setSize(FIT_DIMENSION);

        setVisible(false);
    }

    @Override
    protected void initializeTextFields() {
        tfParamQ2 = createTextField(model.getQ2(), "", createFocusListener(model::setQ2));
        tfParamN = createTextField(model.getN(), "", createFocusListener(model::setN));
        tfParamF2 = createTextField(model.getF2(), "", createFocusListener(model::setF2));
        tfParamD = createTextField(model.getDInterface(), "", createFocusListener(model::setD));
        tfParamD2 = createTextField(model.getD2Interface(), "", createFocusListener(model::setD2));
        tfParamF3 = createTextField(model.getF3(), "", createFocusListener(model::setF3));
        tfParamD3 = createTextField(model.getD3Interface(), "", createFocusListener(model::setD3));
        tfParamQ3 = createTextField(model.getQ3(), "", createFocusListener(model::setQ3));
        tfParamVx = createTextField(model.getVxInterface(), "", createFocusListener(model::setVx));
        tfParamVy = createTextField(model.getVyInterface(), "", createFocusListener(model::setVy));
        tfParamG = createTextField(model.getG(), "", createFocusListener(model::setG));
        tfParamFTrip = createTextField(model.getFTrip(), "", createFocusListener(model::setFTrip));
        tfParamTTrip = createTextField(model.getTTripInterface(), "", createFocusListener(model::setTTrip));
        tfFitStart = createTextField(model.getFitStart(), "", createFocusListener(model::setFitStart));
        // TODO: set this value to channel number
        tfFitEnd = createTextField(model.getFitEnd(), "", createFocusListener(model::setFitEnd));

        // TODO: set this value to the model currently selected
        //  (we can probably just delete it as it duplicates the model name)
        tfFitModel = createTextField("Model", "");
        tfFitModel.setEditable(false);

        tfModProb1 = createTextField(model.getModProb1(), "", createFocusListener(model::setModProb1));
        tfModProb2 = createTextField(model.getModProb2(), "", createFocusListener(model::setModProb2));
        tfModProb3 = createTextField(model.getModProb3(), "", createFocusListener(model::setModProb3));
    }

    private JRadioButton createHoldButton(FitModel.Parameter parameter) {
        JRadioButton radioButton = new JRadioButton("Hold");
        radioButton.setSelected(parameter.isHeld());

        radioButton.addActionListener(ev -> {
            parameter.setHold(radioButton.isSelected());
        });

        return radioButton;
    }

    @Override
    protected void initializeButtons() {
        btnTest = createJButton("Test", "", null, (ItemListener) null);
        btnSetPar = createJButton("Default", "", null, controller.btnResetParametersPressed());
        btnCNNImage = createJButton("ImFCSNet", "", null, (ItemListener) null);
        btnCNNACF = createJButton("FCSNet", "", null, (ItemListener) null);

        tbGLS = createJToggleButton("GLS", "", null, controller.tbOptionPressed(model::setGLS));
        tbGLS.setForeground(Color.LIGHT_GRAY);
        tbBayes = createJToggleButton("Bayes", "", null, controller.tbOptionPressed(model::setBayes));
        tbBayes.setForeground(Color.LIGHT_GRAY);
        tbFixPar = createJToggleButton("Free", "", null, controller.tbFixParPressed());

        holdN = createHoldButton(model.getN());
        holdF2 = createHoldButton(model.getF2());
        holdD = createHoldButton(model.getD());
        holdD2 = createHoldButton(model.getD2());
        holdF3 = createHoldButton(model.getF3());
        holdD3 = createHoldButton(model.getD3());
        holdVx = createHoldButton(model.getVx());
        holdVy = createHoldButton(model.getVy());
        holdG = createHoldButton(model.getG());
        holdFTrip = createHoldButton(model.getFTrip());
        holdTTtrip = createHoldButton(model.getTTrip());

        rbtnCNNImage = new JRadioButton("ImFCSNet");
        rbtnCNNImage.setVisible(false);
        rbtnCNNACF = new JRadioButton("FCSNet");
        rbtnCNNACF.setVisible(false);
    }

    // this method is called after the model has reset default value
    public void setDefaultValues() {
        setText(tfParamQ2, model.getQ2());
        setText(tfParamN, model.getN());
        setText(tfParamF2, model.getF2());
        setText(tfParamD, model.getDInterface());
        setText(tfParamD2, model.getD2Interface());
        setText(tfParamF3, model.getF3());
        setText(tfParamD3, model.getD3Interface());
        setText(tfParamQ3, model.getQ3());
        setText(tfParamVx, model.getVxInterface());
        setText(tfParamVy, model.getVyInterface());
        setText(tfParamG, model.getG());
        setText(tfParamFTrip, model.getFTrip());
        setText(tfParamTTrip, model.getTTripInterface());
        setText(tfFitStart, model.getFitStart());
        setText(tfModProb1, model.getModProb1());
        setText(tfModProb2, model.getModProb2());
        setText(tfModProb3, model.getModProb3());
    }

    @Override
    protected void addComponentsToFrame() {
        // row 1
        add(createJLabel("Fit Model: ", ""));
        add(tfFitModel);
        add(createJLabel("", ""));
        add(tbGLS);
        add(createJLabel("", ""));
        add(tbBayes);

        // row 2 (empty)
        for (int i = 0; i < FIT_LAYOUT.getColumns(); i++) {
            add(createJLabel("", ""));
        }

        // row 3
        for (int i = 0; i < FIT_LAYOUT.getColumns() - 1; i++) {
            add(createJLabel("", ""));
        }
        add(btnTest);

        // row 4
        add(createJLabel("N: ", ""));
        add(tfParamN);
        add(holdN);
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(btnSetPar);

        // row 5
        add(createJLabel("D [μm2/s]: ", ""));
        add(tfParamD);
        add(holdD);
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(tbFixPar);

        // row 6
        add(createJLabel("D2 [μm2/s]", ""));
        add(tfParamD2);
        add(holdD2);
        add(createJLabel("D3: ", ""));
        add(tfParamD3);
        add(holdD3);

        // row 7
        add(createJLabel("F2: ", ""));
        add(tfParamF2);
        add(holdF2);
        add(createJLabel("F3: ", ""));
        add(tfParamF3);
        add(holdF3);

        // row 8
        add(createJLabel("FTrip [μs]: ", ""));
        add(tfParamFTrip);
        add(holdFTrip);
        add(createJLabel("vx [μm/s]: ", ""));
        add(tfParamVx);
        add(holdVx);

        // row 9
        add(createJLabel("TTrip [μs]: ", ""));
        add(tfParamTTrip);
        add(holdTTtrip);
        add(createJLabel("vy [μm/s]: ", ""));
        add(tfParamVy);
        add(holdVy);

        // row 10
        add(createJLabel("G: ", ""));
        add(tfParamG);
        add(holdG);
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));

        // row 11 (empty)
        for (int i = 0; i < FIT_LAYOUT.getColumns(); i++) {
            add(createJLabel("", ""));
        }

        // row 12
        add(createJLabel("Q2: ", ""));
        add(tfParamQ2);
        add(createJLabel("", ""));
        add(createJLabel("Q3: ", ""));
        add(tfParamQ3);
        add(createJLabel("", ""));

        // row 13
        add(createJLabel("Fit start: ", ""));
        add(tfFitStart);
        add(createJLabel("", ""));
        add(createJLabel("Fit end: ", ""));
        add(tfFitEnd);
        add(createJLabel("", ""));

        // row 14 (empty)
        for (int i = 0; i < FIT_LAYOUT.getColumns(); i++) {
            add(createJLabel("", ""));
        }

        // row 15
        add(createJLabel("Bayesian", ""));
        add(createJLabel("Model", ""));
        add(createJLabel("Prob.", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(createJLabel("", ""));

        // row 16
        add(createJLabel("Model 1", ""));
        add(tfModProb1);
        add(createJLabel("Model 2", ""));
        add(tfModProb2);
        add(createJLabel("Model 3", ""));
        add(tfModProb3);

        // row 17 - CNN related settings
        add(btnCNNImage);
        add(rbtnCNNImage);
        add(createJLabel("", ""));
        add(createJLabel("", ""));
        add(btnCNNACF);
        add(rbtnCNNACF);
    }
}
