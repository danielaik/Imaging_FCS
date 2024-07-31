package fiji.plugin.imaging_fcs.new_imfcs.utils;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;

import javax.swing.*;
import java.io.File;
import java.util.Map;

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
