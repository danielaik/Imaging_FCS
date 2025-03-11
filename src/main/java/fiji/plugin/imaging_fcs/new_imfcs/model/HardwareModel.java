package fiji.plugin.imaging_fcs.new_imfcs.model;

import fiji.plugin.imaging_fcs.new_imfcs.utils.CheckOS;

import java.io.*;
import java.nio.file.Files;

import static fiji.plugin.imaging_fcs.gpufit.Gpufit.isCudaAvailable;

/**
 * Represents the hardware configuration model for the Imaging FCS application.
 * This class is responsible for determining if CUDA (GPU computing) is available
 * and fetching any relevant messages regarding the CUDA setup.
 */
public final class HardwareModel {
    private final boolean cuda;
    private String cudaMessage = "CUDA and libs loaded.";

    /**
     * Initializes the hardware model and checks if CUDA is available.
     * Sets the CUDA message accordingly.
     */
    public HardwareModel() {
        this.cuda = loadIfCudaAvailable();
    }

    /**
     * Loads the required GPU libraries based on the operating system.
     * Throws an exception if the OS is not supported or if the libraries cannot be loaded.
     *
     * @throws IOException if loading GPU libraries fails.
     */
    private static void loadGpuLibraries() throws IOException {
        CheckOS.OperatingSystem osType = CheckOS.getCurrentOS();
        if (osType == CheckOS.OperatingSystem.OTHER || osType == CheckOS.OperatingSystem.MAC) {
            throw new RuntimeException("GPU mode is currently supported only on Windows and Linux with NVIDIA GPU.");
        }

        String libName = (osType == CheckOS.OperatingSystem.WINDOWS) ? "agpufit.dll" : "libagpufit.so";

        // Create a temporary directory for the libraries
        File tmpDir = Files.createTempDirectory("gpufitImFCS-lib").toFile();
        // Mark for deletion on exit
        tmpDir.deleteOnExit();

        // Write and load your custom GPU library
        writeLibraryFile(tmpDir, libName);
        System.load(tmpDir + "/" + libName); // Load the custom GPU library
    }

    /**
     * Writes the GPU library file to the specified directory.
     *
     * @param directory the directory to write the library file to.
     * @param libName   the name of the library to write.
     * @throws IOException if writing the library file fails.
     */
    private static void writeLibraryFile(File directory, String libName) throws IOException {
        try (InputStream in = HardwareModel.class.getResourceAsStream(
                "/libs/" + libName); FileOutputStream out = new FileOutputStream(new File(directory, libName))) {

            if (in == null) {
                throw new FileNotFoundException("Library " + libName + " not found in resources");
            }

            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }

    /**
     * Checks if CUDA is available by loading GPU libraries and calling the native method.
     *
     * @return true if CUDA is available, false otherwise.
     */
    private boolean loadIfCudaAvailable() {
        try {
            loadGpuLibraries();
            return isCudaAvailable();
        } catch (Exception e) {
            cudaMessage = e.getMessage();
            return false;
        }
    }

    /**
     * Returns the CUDA availability status.
     *
     * @return true if CUDA is available, false otherwise.
     */
    public boolean isCuda() {
        return cuda;
    }

    /**
     * Returns the CUDA related message or alert.
     *
     * @return A string containing the CUDA message or alert.
     */
    public String getCudaMessage() {
        return cudaMessage;
    }
}