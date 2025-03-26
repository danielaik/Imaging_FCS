package fiji.plugin.imaging_fcs.imfcs.utils;

import fiji.plugin.imaging_fcs.imfcs.enums.DccfDirection;
import fiji.plugin.imaging_fcs.imfcs.enums.FitFunctions;
import fiji.plugin.imaging_fcs.imfcs.model.DiffusionLawModel;
import fiji.plugin.imaging_fcs.imfcs.model.ExpSettingsModel;
import fiji.plugin.imaging_fcs.imfcs.model.PixelModel;
import fiji.plugin.imaging_fcs.imfcs.model.correlations.Correlator;
import ij.IJ;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.streaming.SXSSFWorkbook;

import javax.swing.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
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

        int rowIndex = 0;

        // Here x and y are inverted to follow the behavior of previous ImagingFCS version
        for (int y = 0; y < numCol; y++) {
            for (int x = 0; x < numRow; x++) {
                PixelModel pixelModel = pixelModels[x][y];
                if (pixelModel != null) {
                    double[] array = arrayGetter.apply(pixelModel);
                    if (array == null) {
                        continue;
                    }
                    Row row = sheet.createRow(rowIndex);
                    row.createCell(0).setCellValue(String.format("(%d, %d)", x, y));

                    for (int k = 0; k < array.length; k++) {
                        row.createCell(k + 1).setCellValue(array[k]);
                    }

                    rowIndex++;
                }
            }
        }

        // delete sheet if no data was added
        if (rowIndex == 0) {
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
     * The sheet includes rows for each pixel's position and columns for each fit parameter.
     * Pixel positions are represented as "(x, y)" and fit parameters are listed by name.
     *
     * @param workbook    the workbook to create the sheet in
     * @param name        the name of the sheet
     * @param pixelModels a 2D array of PixelModel objects containing fit parameters
     */
    public static void createFitParametersSheet(Workbook workbook, String name, PixelModel[][] pixelModels) {
        Sheet sheet = workbook.createSheet(name + " - Fit Parameters");
        Row row = sheet.createRow(0);

        row.createCell(0).setCellValue("Coordinate");

        int numParams = PixelModel.paramsName.length;

        // Write parameter names as column headers (starting from column 1)
        for (int i = 0; i < numParams; i++) {
            row.createCell(i + 1).setCellValue(PixelModel.paramsName[i]);
        }

        int numRow = pixelModels.length;
        int numCol = pixelModels[0].length;

        int rowIndex = 1;

        // Here x and y are inverted to follow the behavior of previous ImagingFCS version
        for (int y = 0; y < numCol; y++) {
            for (int x = 0; x < numRow; x++) {
                PixelModel pixelModel = pixelModels[x][y];
                if (pixelModel == null || !pixelModel.isFitted()) {
                    continue;
                }

                row = sheet.createRow(rowIndex);
                row.createCell(0).setCellValue(String.format("(%d, %d)", x, y));

                Pair<String, Double>[] fitParams = pixelModel.getParams();

                for (int i = 0; i < numParams; i++) {
                    row.createCell(i + 1).setCellValue(fitParams[i].getRight());
                }
                rowIndex++;
            }
        }

        // delete sheet if no data was added
        if (rowIndex == 1) {
            workbook.removeSheetAt(workbook.getSheetIndex(sheet));
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
     * Creates a sheet in the workbook to store diffusion law data.
     * Adds data for effective area (Aeff), time, standard deviation (SD), and fit parameters such as intercept, slope,
     * fit start, and fit end.
     *
     * @param workbook          the workbook to add the sheet to
     * @param diffusionLawModel the model containing the diffusion law data
     */
    public static void saveDiffusionLawSheet(Workbook workbook, DiffusionLawModel diffusionLawModel) {
        // check if the diffusion law was computed.
        if (diffusionLawModel.getTime() == null) {
            return;
        }

        Sheet sheet = workbook.createSheet("Diffusion law");
        Row headerRow = sheet.createRow(0);
        headerRow.createCell(0).setCellValue("Aeff");
        headerRow.createCell(1).setCellValue("Time");
        headerRow.createCell(2).setCellValue("SD");
        headerRow.createCell(3).setCellValue("intercept");
        headerRow.createCell(4).setCellValue("slope");
        headerRow.createCell(5).setCellValue("fit start");
        headerRow.createCell(6).setCellValue("fit end");

        for (int i = 0; i < diffusionLawModel.getEffectiveArea().length; i++) {
            Row row = sheet.createRow(i + 1);

            row.createCell(0).setCellValue(diffusionLawModel.getEffectiveArea()[i]);
            row.createCell(1).setCellValue(diffusionLawModel.getTime()[i]);
            row.createCell(2).setCellValue(diffusionLawModel.getStandardDeviation()[i]);
        }

        Row firstRow = sheet.getRow(1);
        firstRow.createCell(3).setCellValue(diffusionLawModel.getIntercept());
        firstRow.createCell(4).setCellValue(diffusionLawModel.getSlope());
        firstRow.createCell(5).setCellValue(diffusionLawModel.getFitStart());
        firstRow.createCell(6).setCellValue(diffusionLawModel.getFitEnd());
    }

    /**
     * Creates a sheet in the workbook to store point spread function (PSF) results.
     * The sheet includes binning values, D values, and their corresponding standard deviations.
     *
     * @param workbook   the workbook to add the sheet to
     * @param psfResults a map where the key is the PSF value and the value is a 2D array containing the results
     */
    public static void savePSFSheet(Workbook workbook, Map<Double, double[][]> psfResults) {
        if (psfResults == null) {
            return;
        }

        Sheet sheet = workbook.createSheet("PSF");
        int column = 0;

        Row row = sheet.createRow(0);
        row.createCell(column).setCellValue("Binning");

        boolean init = false;

        for (Map.Entry<Double, double[][]> entry : psfResults.entrySet()) {
            if (!init) {
                double[] binnings = entry.getValue()[0];

                for (int i = 0; i < binnings.length; i++) {
                    sheet.createRow(i + 1).createCell(column).setCellValue(binnings[i]);
                }

                column += 1;
                init = true;
            }

            row = sheet.getRow(0);
            row.createCell(column).setCellValue(String.format("D (PSF = %.2f)", entry.getKey()));
            row.createCell(column + 1).setCellValue(String.format("SD (PSF = %.2f)", entry.getKey()));

            for (int i = 0; i < entry.getValue()[1].length; i++) {
                row = sheet.getRow(i + 1);
                row.createCell(column).setCellValue(entry.getValue()[1][i]);
                row.createCell(column + 1).setCellValue(entry.getValue()[2][i]);
            }

            column += 2;
        }
    }

    /**
     * Converts a direction name to a safe format for use as an Excel sheet name.
     * Excel does not allow certain characters like "/" or "\", so known constants are replaced
     * with "diagonal up" or "diagonal down".
     *
     * @param direction the original direction enum
     * @return a safe, Excel-compatible direction name
     */
    private static String createSafeDirectionName(DccfDirection direction) {
        if (direction == DccfDirection.DIAGONAL_UP_DIRECTION) {
            return "diagonal up";
        } else if (direction == DccfDirection.DIAGONAL_DOWN_DIRECTION) {
            return "diagonal down";
        }

        return direction.getDisplayName();
    }

    /**
     * Creates and saves dCCF sheet for each direction in the provided map.
     * The sheet names are made Excel-compatible by replacing unsupported characters in the direction name.
     * Each sheet contains correlation values for the corresponding direction, with rows and columns labeled.
     *
     * @param workbook the workbook to add the sheets to
     * @param dccf     a map where the key is the direction name and the value is a 2D array of correlation data
     */
    public static void savedCCFSheets(Workbook workbook, Map<DccfDirection, double[][]> dccf) {
        if (dccf.isEmpty()) {
            return;
        }

        for (Map.Entry<DccfDirection, double[][]> entry : dccf.entrySet()) {
            String directionName = createSafeDirectionName(entry.getKey());
            Sheet sheet = workbook.createSheet("dCCF - " + directionName);
            Row headerRow = sheet.createRow(0);
            headerRow.createCell(0).setCellValue("y / x");


            double[][] values = entry.getValue();
            int numRow = values.length;
            int numCol = values[0].length;

            boolean init = false;

            for (int y = 0; y < numCol; y++) {
                Row row = sheet.createRow(y + 1);
                row.createCell(0).setCellValue(y);
                for (int x = 0; x < numRow; x++) {
                    if (!init) {
                        headerRow.createCell(x + 1).setCellValue(x);
                    }

                    row.createCell(x + 1).setCellValue(values[x][y]);
                }
                init = true;
            }
        }
    }

    /**
     * Creates and saves an Excel sheet with Number and Brightness (N&B) data.
     * The sheet includes multiple sections such as "Number", "Brightness", "Num (corrected)",
     * and "Epsilon (corrected)", with each section containing the values from the corresponding 2D array.
     *
     * @param workbook the workbook to add the sheet to
     * @param NBB      a 2D array representing brightness values
     * @param NBN      a 2D array representing number values
     */
    public static void saveNumberAndBrightnessSheet(Workbook workbook, double[][] NBB, double[][] NBN) {
        if (NBB == null || NBN == null) {
            return;
        }

        int numRows = NBB.length;
        int numCols = NBB[0].length;

        Sheet sheet = workbook.createSheet("N&B");

        // Create the header row.
        Row header = sheet.createRow(0);
        int colIndex = 0;
        header.createCell(colIndex++).setCellValue("Coordinate");
        header.createCell(colIndex++).setCellValue("Number");
        header.createCell(colIndex).setCellValue("Brightness");

        // Now write each data row in full.
        int rowIndex = 1;
        for (int y = 0; y < numCols; y++) {
            for (int x = 0; x < numRows; x++) {
                Row row = sheet.createRow(rowIndex++);
                colIndex = 0;
                // Write coordinate in the first column.
                row.createCell(colIndex++).setCellValue(String.format("(%d, %d)", x, y));
                // Write each additional value if the corresponding array is not null.
                row.createCell(colIndex++).setCellValue(NBN[x][y]);
                row.createCell(colIndex).setCellValue(NBB[x][y]);
            }
        }
    }

    /**
     * Creates sheets in the workbook for the correlation function (CF), standard deviation, fit functions,
     * and MSD data based on the pixel models. Also adds sheets for ACF1 and ACF2 if applicable.
     *
     * @param workbook    the workbook to add the sheets to
     * @param pixelModels the 2D array of PixelModel objects
     * @param settings    the experimental settings used to determine which sheets to create
     * @param correlator  the correlator providing lag and sample times
     */
    public static void saveExcelPixelModels(Workbook workbook, PixelModel[][] pixelModels, ExpSettingsModel settings,
                                            Correlator correlator) {
        if (pixelModels != null) {
            ExcelExporter.createSheetLagTime(workbook, correlator.getLagTimes(), correlator.getSampleTimes());

            saveSheetsPixelModels(workbook, "CF", pixelModels, settings.isMSD());
            if (settings.getFitModel() == FitFunctions.DC_FCCS_2D) {
                saveSheetsPixelModels(workbook, "ACF1",
                        PixelModel.extractAcfPixelModels(pixelModels, PixelModel::getAcf1PixelModel), settings.isMSD());

                saveSheetsPixelModels(workbook, "ACF2",
                        PixelModel.extractAcfPixelModels(pixelModels, PixelModel::getAcf2PixelModel), settings.isMSD());
            }
        }
    }

    /**
     * Saves experimental settings and additional data sheets into an Excel file.
     * The method first writes the experimental settings and then uses the provided addSheets function
     * to add specific data sheets (e.g., CF, ACF, PSF) before saving the file.
     *
     * @param filePath    the path where the Excel file will be saved
     * @param settingsMap a map of experimental settings to be written to the file
     * @param addSheets   a consumer function that adds additional sheets to the workbook
     */
    public static void saveExcelFile(String filePath, Map<String, Object> settingsMap, Consumer<Workbook> addSheets) {
        try (Workbook workbook = new SXSSFWorkbook(100)) {
            // Add different sheets
            ExcelExporter.createSheetFromMap(workbook, "Experimental settings", settingsMap);
            addSheets.accept(workbook);

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
