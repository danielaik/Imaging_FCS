package fiji.plugin.imaging_fcs.imfcs.controller;

/**
 * Exception thrown when a reset operation is rejected by the user.
 * This exception is used to indicate that the user has chosen not to proceed
 * with resetting the results, allowing the application to handle this scenario
 * appropriately.
 */
public class RejectResetException extends RuntimeException {
    /**
     * Constructs a new RejectResetException with no detail message.
     */
    public RejectResetException() {
        super();
    }
}
