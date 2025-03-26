package fiji.plugin.imaging_fcs.imfcs.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;

/**
 * Utility class to extract and load native libraries from jar resources.
 * <p>
 * This class provides methods to load native libraries either individually or in batches.
 * For system-specific libraries, it first attempts to load from system paths, falling back
 * to extracting from jar resources if needed.
 * </p>
 */
public final class LibraryLoader {

    // Prevent instantiation.
    private LibraryLoader() {
    }

    /**
     * Loads a single native library from jar resources.
     * <p>
     * This method extracts the library into the given temporary directory and loads it via {@code System.load()}.
     * </p>
     *
     * @param resourceDir the directory in the jar where the library is located.
     *                    For example, "/libs/andor/".
     * @param libName     the name of the native library file to load, e.g. "atmcd64d.dll".
     * @param tmpDir      the temporary directory in which to extract the library.
     * @throws IOException if an I/O error occurs during extraction or if the library is not found.
     */
    private static void loadNativeLibraryFromDirectory(String resourceDir, String libName,
                                                       File tmpDir) throws IOException {
        // Ensure resourceDir ends with a slash
        if (!resourceDir.endsWith("/")) {
            resourceDir += "/";
        }

        // Extract the library file from the jar resources.
        File extractedLib = extractLibraryFile(tmpDir, resourceDir, libName);
        // Load the native library.
        System.load(extractedLib.getAbsolutePath());
    }

    /**
     * Loads multiple native libraries from the same resource directory.
     * <p>
     * This method creates a single temporary directory for all extractions.
     * </p>
     *
     * @param resourceDir the resource directory (ending with a slash, e.g. "/libs/andor/")
     * @param libNames    an array of library file names to load.
     * @throws IOException if any library fails to load.
     */
    public static void loadNativeLibraries(String resourceDir, String... libNames) throws IOException {
        File tmpDir = Files.createTempDirectory("nativeLib").toFile();
        tmpDir.deleteOnExit();
        for (String libName : libNames) {
            loadNativeLibraryFromDirectory(resourceDir, libName, tmpDir);
        }
    }

    /**
     * Loads multiple system-specific libraries.
     * <p>
     * This method attempts to load each library from the system paths first.
     * If the library is not found, it falls back to loading it from the bundled jar resources.
     * </p>
     *
     * @param resourceDir the resource directory (e.g. "/libs/camera_readout/sdk2/")
     * @param libNames    an array of library file names to load.
     * @throws IOException if any library fails to load.
     */
    public static void loadSystemLibraries(String resourceDir, String... libNames) throws IOException {
        // Create one temp directory for fallbacks.
        File tmpDir = Files.createTempDirectory("nativeLib").toFile();
        tmpDir.deleteOnExit();
        for (String libName : libNames) {
            try {
                System.loadLibrary(removeExtension(libName));
            } catch (UnsatisfiedLinkError e) {
                loadNativeLibraryFromDirectory(resourceDir, libName, tmpDir);
            }
        }
    }

    /**
     * Extracts a library file from the jar resources and writes it to the specified directory.
     *
     * @param directory   the directory to which the library file will be extracted.
     * @param resourceDir the resource directory path in the jar (ending with a slash), e.g. "/libs/andor/".
     * @param libName     the name of the library file to extract.
     * @return a {@code File} object representing the extracted library file.
     * @throws IOException if the library cannot be found or an I/O error occurs.
     */
    private static File extractLibraryFile(File directory, String resourceDir, String libName) throws IOException {
        File outputFile = new File(directory, libName);
        try (InputStream in = LibraryLoader.class.getResourceAsStream(
                resourceDir + libName); FileOutputStream out = new FileOutputStream(outputFile)) {

            if (in == null) {
                throw new IOException("Library " + libName + " not found in resources: " + resourceDir);
            }

            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
        return outputFile;
    }

    /**
     * Removes the file extension from a library name.
     * <p>
     * For example, "atmcd64d.dll" becomes "atmcd64d".
     * </p>
     *
     * @param libName the full library file name.
     * @return the library name without its extension.
     */
    private static String removeExtension(String libName) {
        int dotIndex = libName.lastIndexOf('.');
        return (dotIndex > 0) ? libName.substring(0, dotIndex) : libName;
    }
}