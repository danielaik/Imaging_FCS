package fiji.plugin.imaging_fcs.new_imfcs.utils;

import fiji.plugin.imaging_fcs.new_imfcs.constants.Constants;
import fiji.plugin.imaging_fcs.new_imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import fiji.plugin.imaging_fcs.new_imfcs.model.correlations.Correlator;
import ij.IJ;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import javax.swing.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * Utility class for exporting data to Excel files.
 */
public final class ExcelExporter {
    // Private constructor to prevent instantiation
    private ExcelExporter() {
    }

    /**
     * Creates a sheet in the given workbook from a map of data.
     * Each entry in the map is written to a new row in the sheet, with the key in the first column and the value in
     * the second column.
     *
     * @param workbook  the workbook to create the sheet in
     * @param sheetName the name of the sheet to be created
     * @param data      the map containing the data to be written to the sheet
     */
    public static void createSheetFromMap(Workbook workbook, String sheetName, Map<String, Object> data) {
        Sheet sheet = workbook.createSheet(sheetName);
        Row headerRow = sheet.createRow(0);
        headerRow.createCell(0).setCellValue("Parameter Name");
        headerRow.createCell(1).setCellValue("Parameter Value");

        int rowNum = 1;
        for (Map.Entry<String, Object> entry : data.entrySet()) {
            Row row = sheet.createRow(rowNum++);
            row.createCell(0).setCellValue(entry.getKey());
            row.createCell(1).setCellValue(entry.getValue().toString());
        }
    }

    /**
     * Retrieves the row at the specified index from the given sheet.
     * If the row does not exist, a new row is created at that index.
     *
     * @param sheet the sheet from which to retrieve or create the row
     * @param i     the index of the row to retrieve or create
     * @return the existing or newly created row at the specified index
     */
    private static Row getOrCreateRow(Sheet sheet, int i) {
        Row row = sheet.getRow(i);

        if (row == null) {
            row = sheet.createRow(i);
        }

        return row;
    }

    /**
     * Creates a sheet in the given workbook from a 2D array of PixelModel objects.
     * The data for each PixelModel is extracted using the provided array getter function
     * and written to the sheet in a specific format.
     *
     * @param workbook    the workbook to create the sheet in
     * @param sheetName   the name of the sheet to be created
     * @param pixelModels a 2D array of PixelModel objects containing the data to be written to the sheet
     * @param arrayGetter a function that extracts an array of doubles from a PixelModel object
     */
    public static void createSheetFromPixelModelArray(Workbook workbook, String sheetName, PixelModel[][] pixelModels,
                                                      Function<PixelModel, double[]> arrayGetter) {
        Sheet sheet = workbook.createSheet(sheetName);
        int numRow = pixelModels.length;
        int numCol = pixelModels[0].length;

        int columnIndex = 0;

        // Here x and y are inverted to follow the behavior of previous ImagingFCS version
        for (int y = 0; y < numCol; y++) {
            for (int x = 0; x < numRow; x++) {
                PixelModel pixelModel = pixelModels[x][y];
                if (pixelModel != null) {
                    double[] array = arrayGetter.apply(pixelModel);
                    if (array == null) {
                        continue;
                    }
                    Row row = getOrCreateRow(sheet, 0);
                    row.createCell(columnIndex).setCellValue(String.format("(%d, %d)", x, y));

                    for (int k = 0; k < array.length; k++) {
                        row = getOrCreateRow(sheet, k + 1);
                        row.createCell(columnIndex).setCellValue(array[k]);
                    }

                    columnIndex++;
                }
            }
        }

        if (columnIndex == 0) {
            // delete sheet if no data was added
            workbook.removeSheetAt(workbook.getSheetIndex(sheet));
        }
    }

    /**
     * Creates a sheet in the given workbook to display lag times and sample times.
     * The sheet contains three columns: S/N, lag time, and bin width.
     *
     * @param workbook    the workbook to create the sheet in
     * @param lagTimes    an array of lag times to be written to the sheet
     * @param sampleTimes an array of sample times (bin widths) corresponding to each lag time
     */
    public static void createSheetLagTime(Workbook workbook, double[] lagTimes, int[] sampleTimes) {
        Sheet sheet = workbook.createSheet("Lag Time");
        Row headerRow = sheet.createRow(0);
        headerRow.createCell(0).setCellValue("S/N");
        headerRow.createCell(1).setCellValue("LagTime");
        headerRow.createCell(2).setCellValue("Bin width");

        for (int i = 0; i < lagTimes.length; i++) {
            Row row = sheet.createRow(i + 1);
            row.createCell(0).setCellValue(i);
            row.createCell(1).setCellValue(lagTimes[i]);
            row.createCell(2).setCellValue(sampleTimes[i]);
        }
    }

    /**
     * Creates a sheet in the given workbook to display fit parameters for each pixel model.
     * The sheet includes columns for each pixel's position and rows for each fit parameter.
     * Pixel positions are represented as "(x, y)" and fit parameters are listed by name.
     *
     * @param workbook    the workbook to create the sheet in
     * @param name        the name of the sheet
     * @param pixelModels a 2D array of PixelModel objects containing fit parameters
     */
    public static void createFitParametersSheet(Workbook workbook, String name, PixelModel[][] pixelModels) {
        Sheet sheet = workbook.createSheet(name + " - Fit Parameters");
        Row row = sheet.createRow(0);

        row.createCell(0).setCellValue("Parameter");

        int numParams = PixelModel.paramsName.length;

        for (int i = 0; i < numParams; i++) {
            row = sheet.createRow(i + 1);
            row.createCell(0).setCellValue(PixelModel.paramsName[i]);
        }

        int numRow = pixelModels.length;
        int numCol = pixelModels[0].length;

        int columnIndex = 1;

        // Here x and y are inverted to follow the behavior of previous ImagingFCS version
        for (int y = 0; y < numCol; y++) {
            for (int x = 0; x < numRow; x++) {
                PixelModel pixelModel = pixelModels[x][y];
                if (pixelModel == null || !pixelModel.isFitted()) {
                    continue;
                }

                row = sheet.getRow(0);
                row.createCell(columnIndex).setCellValue(String.format("(%d, %d)", x, y));

                Pair<String, Double>[] fitParams = pixelModel.getParams();

                for (int i = 0; i < numParams; i++) {
                    row = sheet.getRow(i + 1);
                    row.createCell(columnIndex).setCellValue(fitParams[i].getRight());
                }
                columnIndex++;
            }
        }
    }

    /**
     * Checks if a file at the given path already exists. If it does, prompts the user to confirm whether they want
     * to replace the file.
     * If the user chooses not to replace the file, null is returned. Otherwise, the original path is returned.
     *
     * @param path the path of the file to check
     * @return the original path if the file does not exist or the user chooses to replace it, null otherwise
     */
    private static String checkIfFileAlreadyExists(String path) {
        File file = new File(path);
        if (file.exists()) {
            int confirm = JOptionPane.showConfirmDialog(null, "The file already exists. Do you want to replace it?",
                    "File already exists", JOptionPane.YES_NO_OPTION);
            if (confirm != JOptionPane.YES_OPTION) {
                return null;
            }
        }

        return path;
    }

    /**
     * Opens a file chooser dialog for the user to select a location and name for saving an Excel file.
     * The default file name is provided as a suggestion, and the selected file path is returned.
     * If the file already exists, the user is prompted to confirm if they want to replace it.
     * If the user cancels the operation, null is returned.
     *
     * @param defaultName the suggested default file name
     * @param openPath    the initial directory path where the file chooser opens
     * @return the selected file path, or null if the user cancels the operation or chooses not to replace an
     * existing file
     */
    public static String selectExcelFileToSave(String defaultName, String openPath) {
        JFileChooser fileChooser = new JFileChooser(openPath);
        fileChooser.setDialogTitle("Specify a file to save");
        fileChooser.setSelectedFile(new File(defaultName));

        int userSelection = fileChooser.showSaveDialog(null);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fileChooser.getSelectedFile();
            String filePath = fileToSave.getAbsolutePath();
            if (!filePath.endsWith(".xlsx")) {
                filePath += ".xlsx";
            }

            return checkIfFileAlreadyExists(filePath);
        }

        return null;
    }

    /**
     * Saves multiple pixel model sheets into the workbook if all models are not null.
     *
     * @param workbook    the workbook to add the sheets to
     * @param name        the base name for the sheets
     * @param pixelModels the 2D array of PixelModel objects
     * @param isMSD       boolean flag indicating if MSD data should be included
     */
    private static void saveSheetsPixelModels(Workbook workbook, String name, PixelModel[][] pixelModels,
                                              boolean isMSD) {
        boolean allNull = Arrays.stream(pixelModels).flatMap(Arrays::stream).allMatch(Objects::isNull);
        if (allNull) {
            IJ.log("All pixel models are null for " + name);
            return;
        }

        ExcelExporter.createSheetFromPixelModelArray(workbook, name, pixelModels, PixelModel::getCorrelationFunction);
        ExcelExporter.createSheetFromPixelModelArray(workbook, name + " - Standard Deviation", pixelModels,
                PixelModel::getStandardDeviationCF);

        if (PixelModel.anyPixelFit(pixelModels)) {
            ExcelExporter.createSheetFromPixelModelArray(workbook, name + " - Fit Functions", pixelModels,
                    PixelModel::getFittedCF);
            ExcelExporter.createSheetFromPixelModelArray(workbook, name + " - Residuals", pixelModels,
                    PixelModel::getResiduals);
            ExcelExporter.createFitParametersSheet(workbook, name, pixelModels);
        }

        if (isMSD) {
            ExcelExporter.createSheetFromPixelModelArray(workbook, name + " - MSD", pixelModels, PixelModel::getMSD);
        }
    }

    /**
     * Saves PixelModel data, experimental settings, and correlator information into an Excel file at the given path.
     * Creates various sheets for CF, standard deviation, fit functions, and MSD based on data.
     *
     * @param filePath    the path where the Excel file will be saved
     * @param pixelModels the 2D array of PixelModel objects containing ACF and fit data
     * @param settings    the experimental settings model used for determining whether MSD data should be included
     * @param correlator  the correlator providing lag times and sample times for the data
     * @param settingsMap a map of experimental settings to be exported in a dedicated sheet
     */
    public static void saveExcelFile(String filePath, PixelModel[][] pixelModels, ExpSettingsModel settings,
                                     Correlator correlator, Map<String, Object> settingsMap) {
        try (Workbook workbook = new XSSFWorkbook()) {
            // Add different sheets
            if (pixelModels != null) {
                ExcelExporter.createSheetFromMap(workbook, "Experimental settings", settingsMap);

                ExcelExporter.createSheetLagTime(workbook, correlator.getLagTimes(), correlator.getSampleTimes());

                saveSheetsPixelModels(workbook, "CF", pixelModels, settings.isMSD());
                if (settings.getFitModel().equals(Constants.DC_FCCS_2D)) {
                    saveSheetsPixelModels(workbook, "ACF1",
                            PixelModel.extractAcfPixelModels(pixelModels, PixelModel::getAcf1PixelModel),
                            settings.isMSD());

                    saveSheetsPixelModels(workbook, "ACF2",
                            PixelModel.extractAcfPixelModels(pixelModels, PixelModel::getAcf2PixelModel),
                            settings.isMSD());
                }
            }

            try (FileOutputStream fileOut = new FileOutputStream(filePath)) {
                workbook.write(fileOut);
                IJ.log(String.format("File saved at %s.", filePath));
            } catch (IOException e) {
                IJ.showMessage("Error saving result table", e.getMessage());
            }
        } catch (IOException e) {
            IJ.showMessage("Error writing result table", e.getMessage());
        }
    }
}
