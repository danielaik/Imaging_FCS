package fiji.plugin.imaging_fcs.new_imfcs.utils;

import java.util.Arrays;

/**
 * Utility class for deep copying 2D arrays.
 * This class provides a method to create a deep copy of a given 2D double array.
 * The class is designed to be non-instantiable by having a private constructor.
 */
public final class MatrixDeepCopy {
    /**
     * Private constructor to prevent instantiation of this utility class.
     */
    private MatrixDeepCopy() {
    }

    /**
     * Creates a deep copy of a 2D double array.
     *
     * @param array the 2D double array to be copied
     * @return a deep copy of the input array, or null if the input array is null
     */
    public static double[][] deepCopy(double[][] array) {
        if (array == null) {
            return null;
        }

        double[][] copy = new double[array.length][];
        for (int i = 0; i < array.length; i++) {
            copy[i] = Arrays.copyOf(array[i], array[i].length);
        }
        return copy;
    }
}
