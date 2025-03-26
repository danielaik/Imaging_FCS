package fiji.plugin.imaging_fcs.imfcs.enums;

public final class EnumUtils {
    private EnumUtils() {
        // utility class, no instances
    }

    /**
     * Converts a display name string into the corresponding enum constant,
     * case-insensitive.
     *
     * @param enumClass The enum’s .class object (e.g. FilterMode.class).
     * @param text      The display name to parse.
     * @param <E>       The enum type, which must implement DisplayNameEnum.
     * @return The matching enum constant.
     * @throws IllegalArgumentException if no match is found.
     */
    public static <E extends Enum<E> & DisplayNameEnum> E fromDisplayName(Class<E> enumClass, String text) {
        for (E constant : enumClass.getEnumConstants()) {
            if (constant.getDisplayName().equalsIgnoreCase(text)) {
                return constant;
            }
        }
        throw new IllegalArgumentException("Unknown " + enumClass.getSimpleName() + " value: " + text);
    }
}
