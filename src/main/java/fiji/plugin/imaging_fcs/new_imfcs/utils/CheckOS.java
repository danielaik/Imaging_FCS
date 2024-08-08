package fiji.plugin.imaging_fcs.new_imfcs.utils;

/**
 * Helper class to determine the operating system.
 */
public final class CheckOS {
    // Private constructor to prevent instantiation
    private CheckOS() {
    }

    /**
     * Determines the current operating system.
     *
     * @return the current operating system as an OperatingSystem enum value
     */
    public static OperatingSystem getCurrentOS() {
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("win")) {
            return OperatingSystem.WINDOWS;
        } else if (osName.contains("mac")) {
            return OperatingSystem.MAC;
        } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("aix")) {
            return OperatingSystem.LINUX;
        } else {
            return OperatingSystem.OTHER;
        }
    }

    /**
     * Enum representing different operating systems.
     */
    public enum OperatingSystem {
        WINDOWS, MAC, LINUX, OTHER;
    }
}
