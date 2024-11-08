package fiji.plugin.imaging_fcs.new_imfcs.controller;

import fiji.plugin.imaging_fcs.new_imfcs.model.*;
import fiji.plugin.imaging_fcs.new_imfcs.view.dialogs.ParameterVideoView;
import ij.IJ;

import javax.swing.*;
import java.io.File;

/**
 * Controls the creation of a parameter video by linking the model and view.
 */
public class ParameterVideoController {
    private final ParameterVideoModel model;
    private final ImageModel imageModel;

    /**
     * Initializes the controller with the settings, image, and fit models.
     *
     * @param settingsModel the experiment settings model
     * @param imageModel    the image model
     * @param fitModel      the fit model
     */
    public ParameterVideoController(ExpSettingsModel settingsModel, ImageModel imageModel, FitModel fitModel) {
        this.imageModel = imageModel;
        model = new ParameterVideoModel(settingsModel, imageModel, fitModel);
    }

    /**
     * Opens a directory chooser dialog for the user to select a folder.
     * A new folder with the specified name will be created inside the selected directory.
     * If a folder with the same name already exists, it does nothing.
     * If a file with the same name exists, it returns null.
     *
     * @param newFolderName the name of the new folder to create
     * @param openPath      the initial directory path where the chooser opens
     * @return the full path of the created folder, or null if a file with the same name exists
     */
    private static String selectDirectoryAndCreateFolder(String newFolderName, String openPath) {
        // Open a directory chooser for the user
        JFileChooser dirChooser = new JFileChooser(openPath);
        dirChooser.setDialogTitle("Select a Directory");
        dirChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        int userSelection = dirChooser.showOpenDialog(null);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            // Get the selected directory
            File selectedDir = dirChooser.getSelectedFile();
            String selectedDirPath = selectedDir.getAbsolutePath();

            // Create the full path for the new folder
            File newFolder = new File(selectedDirPath, newFolderName);

            // Check if a file with the same name exists
            if (newFolder.exists() && newFolder.isFile()) {
                throw new RuntimeException("A file with the same name as the new folder exists.");
            }

            // If the folder doesn't exist, create it
            if (!newFolder.exists()) {
                // Create the folder
                boolean folderCreated = newFolder.mkdir();
                if (!folderCreated) {
                    throw new RuntimeException("Failed to create the folder: " + newFolder.getAbsolutePath());
                }
            }

            // Return the full path of the created or existing folder
            return newFolder.getAbsolutePath();
        }

        // Return null if the user cancels the operation
        return null;
    }

    /**
     * Displays the parameter video configuration dialog for the given frame range.
     *
     * @param firstFrame the first frame of the video
     * @param lastFrame  the last frame of the video
     */
    public void showParameterVideoView(int firstFrame, int lastFrame) {
        new ParameterVideoView(firstFrame, lastFrame, model, this::createParameterVideo);
    }

    /**
     * Processes the input from the view and updates the model accordingly.
     *
     * @param view the view containing the user input
     */
    private void createParameterVideo(ParameterVideoView view) {
        model.setStart((int) view.getNextNumber());
        model.setEnd((int) view.getNextNumber());
        model.setLength((int) view.getNextNumber());
        model.setStep((int) view.getNextNumber());
        model.setSaveCFAndFitPVideo(view.getNextBoolean());
        model.setVideoName(view.getNextString());

        if (model.isSaveCFAndFitPVideo()) {
            String excelDirectory = selectDirectoryAndCreateFolder(String.format("%s_CFAndFit", model.getVideoName()),
                    imageModel.getImagePath());
            if (excelDirectory == null) {
                IJ.showMessage("Error", "No directory selected.");
                return;
            }

            model.setExcelDirectory(excelDirectory);
        }

        new BackgroundTaskWorker<Void, Void>(() -> {
            try {
                model.createParameterVideo();
            } catch (Exception e) {
                IJ.showMessage(e.getMessage());
            }
        }).execute();
    }
}
