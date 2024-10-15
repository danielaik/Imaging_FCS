package fiji.plugin.imaging_fcs.new_imfcs.utils;

import com.github.pjfanning.xlsx.StreamingReader;
import com.github.pjfanning.xlsx.exceptions.MissingSheetException;
import fiji.plugin.imaging_fcs.new_imfcs.model.PixelModel;
import ij.IJ;
import org.apache.poi.ss.usermodel.*;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.DoubleStream;

/**
 * Utility class for importing data from Excel files.
 */
public final class ExcelReader {
    // Private constructor to prevent instantiation
    private ExcelReader() {
    }

    /**
     * Retrieves the value from a cell and returns it as an Object.
     * Handles different cell types such as STRING, NUMERIC, and BOOLEAN.
     *
     * @param cell the cell to retrieve the value from
     * @return the value of the cell as an Object, or null if the cell type is unsupported
     */
    private static Object getCellValue(Cell cell) {
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                return cell.getNumericCellValue();
            case BOOLEAN:
                return cell.getBooleanCellValue();
            default:
                return null;
        }
    }

    /**
     * Reads data from a sheet and returns it as a map.
     * Assumes the first column contains keys and the second column contains values.
     *
     * @param workbook  the workbook to read from
     * @param sheetName the name of the sheet to read
     * @return a map containing the data from the sheet
     */
    public static Map<String, Object> readSheetToMap(Workbook workbook, String sheetName) {
        Map<String, Object> data = new HashMap<>();

        Sheet sheet = workbook.getSheet(sheetName);

        for (Row row : sheet) {
            // Skip the header row
            if (row.getRowNum() == 0) {
                continue;
            }

            Cell keyCell = row.getCell(0);
            Cell valueCell = row.getCell(1);
            if (keyCell != null && valueCell != null) {
                data.put(keyCell.getStringCellValue(), getCellValue(valueCell));
            }
        }

        return data;
    }

    /**
     * Parses a position string in the format "(x, y)" and returns a Point object.
     *
     * @param position the position string to parse
     * @return a Point object representing the parsed coordinates
     * @throws IllegalArgumentException if the position string is not properly formatted
     */
    private static Point parsePosition(String position) {
        try {
            String[] parts = position.replaceAll("[()]", "").split(",");
            int x = Integer.parseInt(parts[0].trim());
            int y = Integer.parseInt(parts[1].trim());
            return new Point(x, y);
        } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
            throw new IllegalArgumentException("Invalid position format: " + position, e);
        }
    }

    /**
     * Retrieves values from a column in the sheet and returns them as a double array.
     *
     * @param sheet       the sheet containing the data
     * @param columnIndex the index of the column to retrieve
     * @return a double array containing the values from the column
     */
    private static double[] getValues(Sheet sheet, int columnIndex) {
        List<Double> valuesList = new ArrayList<>();

        // Get the row iterator and skip the first row (index 0)
        Iterator<Row> rowIterator = sheet.iterator();
        if (rowIterator.hasNext()) {
            // Skip the header row
            rowIterator.next();
        }

        // Iterate over the remaining rows
        while (rowIterator.hasNext()) {
            Row row = rowIterator.next();
            Cell cell = row.getCell(columnIndex);

            // Stop if the cell is null or not numeric
            if (cell == null || cell.getCellType() != CellType.NUMERIC) {
                break;
            }

            valuesList.add(cell.getNumericCellValue());
        }

        return valuesList.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Extracts numeric values from a row, excluding the first cell.
     *
     * @param row the row to extract values from
     * @return a double array with numeric values from the row
     */
    private static double[] getValuesFromRow(Row row) {
        int numValues = row.getPhysicalNumberOfCells() - 1; // Exclude the coordinate cell
        double[] values = new double[numValues];

        for (int i = 0; i < numValues; i++) {
            values[i] = row.getCell(i + 1).getNumericCellValue();
        }
        return values;
    }

    /**
     * Retrieves or initializes a PixelModel at the specified position in the array.
     *
     * @param pixelModels the 2D array of PixelModel objects
     * @param position    the position to get or initialize the PixelModel
     * @return the existing or newly initialized PixelModel
     */
    private static PixelModel getOrInitPixelModel(PixelModel[][] pixelModels, Point position) {
        if (pixelModels[position.x][position.y] == null) {
            pixelModels[position.x][position.y] = new PixelModel();
        }

        return pixelModels[position.x][position.y];
    }

    /**
     * Reads data from a sheet and updates PixelModel objects in a 2D array.
     * The first row of the sheet contains position strings in the format "(x, y)".
     *
     * @param workbook    the workbook containing the data
     * @param sheetName   the name of the sheet to read
     * @param pixelModels a 2D array of PixelModel objects to update
     * @param setter      a function that sets values in a PixelModel object
     */
    public static void readSheetToPixelModels(Workbook workbook, String sheetName, PixelModel[][] pixelModels,
                                              BiConsumer<PixelModel, double[]> setter) {
        Sheet sheet;
        try {
            sheet = workbook.getSheet(sheetName);
        } catch (MissingSheetException e) {
            return;
        }

        Iterator<Row> rowIterator = sheet.iterator();
        if (!rowIterator.hasNext()) {
            IJ.log(String.format("Sheet '%s' is empty.", sheetName));
            return;
        }

        Row row = rowIterator.next();
        try {
            parsePosition(row.getCell(0).getStringCellValue());
        } catch (IllegalArgumentException e) {
            row = rowIterator.next();
        }

        while (row != null) {
            Point position = parsePosition(row.getCell(0).getStringCellValue());
            PixelModel pixelModel = getOrInitPixelModel(pixelModels, position);
            setter.accept(pixelModel, getValuesFromRow(row));

            row = rowIterator.hasNext() ? rowIterator.next() : null;
        }
    }

    /**
     * Reads lag times and sample times from a sheet and applies them using provided setter functions.
     * Assumes lag times are in the second column and sample times in the third column.
     *
     * @param workbook          the workbook containing the data
     * @param sheetName         the name of the sheet to read
     * @param lagTimesSetter    a consumer function to accept lag times
     * @param sampleTimesSetter a consumer function to accept sample times
     */
    public static void readLagTimesAndSampleTimes(Workbook workbook, String sheetName,
                                                  Consumer<double[]> lagTimesSetter,
                                                  Consumer<int[]> sampleTimesSetter) {
        Sheet sheet;
        try {
            sheet = workbook.getSheet(sheetName);
        } catch (MissingSheetException e) {
            return;
        }

        lagTimesSetter.accept(getValues(sheet, 1));
        sampleTimesSetter.accept(DoubleStream.of(getValues(sheet, 2)).mapToInt(d -> (int) d).toArray());
    }

    /**
     * Opens a file chooser to select an Excel file and returns a Workbook instance.
     *
     * @param openPath the initial directory path for the file chooser
     * @return a Workbook if a file is selected and successfully loaded, or null if the selection is canceled
     * @throws RuntimeException if an error occurs while reading the file
     */
    public static Workbook selectExcelFileToLoad(String openPath) {
        JFileChooser fileChooser = new JFileChooser(openPath);
        fileChooser.setDialogTitle("Open Excel File");
        int userSelection = fileChooser.showOpenDialog(null);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToOpen = fileChooser.getSelectedFile();
            String filePath = fileToOpen.getAbsolutePath();

            try (FileInputStream fileIn = new FileInputStream(filePath)) {
                return StreamingReader.builder().rowCacheSize(100).bufferSize(4096).open(fileIn);
            } catch (IOException e) {
                throw new RuntimeException("Error reading Excel file");
            }
        }

        return null;
    }
}
