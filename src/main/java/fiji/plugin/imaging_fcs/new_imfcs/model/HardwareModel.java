package fiji.plugin.imaging_fcs.new_imfcs.model;

/**
 * Represents the hardware configuration model for the Imaging FCS application.
 * This class is responsible for determining if CUDA (GPU computing) is available
 * and fetching any relevant messages regarding the CUDA setup.
 */
public final class HardwareModel {
    private final boolean cuda;
    private final String cudaMessage;

    /**
     * Constructor for the HardwareModel. It initializes the CUDA availability
     * and the CUDA message based on the GpufitImFCS library's status.
     */
    public HardwareModel() {
        // this.cuda = GpufitImFCS.isCudaAvailable();
        this.cuda = false;
        // this.cudaMessage = GpufitImFCS.ALERT;
        this.cudaMessage = "BETA version: you may experience some bugs.";
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
