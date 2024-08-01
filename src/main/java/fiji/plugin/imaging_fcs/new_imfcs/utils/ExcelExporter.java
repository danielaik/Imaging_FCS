package fiji.plugin.imaging_fcs.new_imfcs.utils;

import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;

import javax.swing.*;
import java.io.File;
import java.util.Map;
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
    public static void createSheetFromPixelModelArray(Workbook workbook, String sheetName, PixelModel[][] pixelModels
            , Function<PixelModel, double[]> arrayGetter) {
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
    }

    /**
     * Opens a file chooser dialog for the user to select a location and name for saving an Excel file.
     * The default file name is provided as a suggestion, and the selected file path is returned.
     * If the user cancels the operation, null is returned.
     *
     * @param defaultName the suggested default file name
     * @return the selected file path, or null if the user cancels the operation
     */
    public static String selectExcelFileToSave(String defaultName) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Specify a file to save");
        fileChooser.setSelectedFile(new File(defaultName));

        int userSelection = fileChooser.showSaveDialog(null);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fileChooser.getSelectedFile();
            String filePath = fileToSave.getAbsolutePath();
            if (!filePath.endsWith(".xlsx")) {
                filePath += ".xlsx";
            }

            return filePath;
        }

        return null;
    }
}
