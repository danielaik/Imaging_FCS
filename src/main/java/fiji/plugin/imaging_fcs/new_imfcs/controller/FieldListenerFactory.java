package fiji.plugin.imaging_fcs.new_imfcs.controller;

import ij.IJ;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.util.function.Consumer;

import static fiji.plugin.imaging_fcs.new_imfcs.controller.ControllerUtils.getComboBoxSelectionFromEvent;

/**
 * A factory class for creating listeners for Swing components, such as JTextField and JComboBox,
 * to handle focus and action events with specific behaviors. This class simplifies the process
 * of attaching listeners that react to changes and validate input, ensuring consistent application
 * behavior.
 */
public final class FieldListenerFactory {
    // Private constructor to prevent instantiation
    private FieldListenerFactory() {
    }

    /**
     * Creates a {@link FocusListener} that performs an action with the text of a {@link JTextField}
     * when the component loses focus. The action is specified by the {@link Consumer<String>} passed
     * to the method.
     *
     * @param setter A {@link Consumer<String>} that defines the action to be performed with the
     *               text value of the {@link JTextField} upon losing focus. This consumer receives
     *               the text value of the text field as its input. If the format is not valid, it restores
     *               the initial value that was in the field on gaining focus.
     * @return A {@link FocusListener} that captures the text of a {@link JTextField} when focus is lost
     * and performs the specified action with it.
     */
    public static FocusListener createFocusListener(Consumer<String> setter) {
        return new FocusListener() {
            // This field will be used to store the current value of the text field
            String memory;

            // Store the value on focus gained
            @Override
            public void focusGained(FocusEvent ev) {
                JTextField textField = (JTextField) ev.getComponent();
                memory = textField.getText();
            }

            @Override
            public void focusLost(FocusEvent ev) {
                JTextField textField = (JTextField) ev.getComponent();

                // remove the extra spaces if there are some and set to lowercase
                String text = textField.getText().toLowerCase().trim().replaceAll("\\s+", " ");
                textField.setText(text);

                try {
                    if (!text.replaceAll("\\s+", "").equals(memory.replaceAll("\\s+", ""))) {
                        setter.accept(text);
                    }
                } catch (InvalidUserInputException e) {
                    IJ.showMessage("Error", e.getMessage());
                    textField.setText(memory);
                } catch (RejectResetException e) {
                    textField.setText(memory);
                } catch (Exception e) {
                    // The text field was not successfully parsed, a message is shown and the value is restored
                    IJ.showMessage("Error", "Incorrect format for this field");
                    textField.setText(memory);
                }
            }
        };
    }

    /**
     * Creates an {@link ActionListener} for a {@link JComboBox} that updates a property based on
     * the selected value. It maintains the previous selection and resets the combo box to the
     * previous value if an exception occurs during the update.
     *
     * @param comboBox The {@link JComboBox} whose selections are being monitored.
     * @param setter   A {@link Consumer<String>} that processes the selected value of the
     *                 {@link JComboBox}.
     * @return An {@link ActionListener} that updates the property via the setter with
     * the current selection of the {@link JComboBox} and handles exceptions by restoring
     * the previous selection.
     */
    public static ActionListener createComboBoxListener(JComboBox<String> comboBox, Consumer<String> setter) {
        return new ActionListener() {
            // This field will be used to store the previous value of the combo box
            private String previousSelection = (String) comboBox.getSelectedItem();

            @Override
            public void actionPerformed(ActionEvent ev) {
                // Get the current selection
                String currentSelection = getComboBoxSelectionFromEvent(ev);

                try {
                    if (!previousSelection.equals(currentSelection)) {
                        // Attempt to update using the setter
                        setter.accept(currentSelection);
                        // Update previous selection to current if successful
                        previousSelection = currentSelection;
                    }
                } catch (RejectResetException e) {
                    comboBox.setSelectedItem(previousSelection);
                }
            }
        };
    }
}