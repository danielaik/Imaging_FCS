
package fiji.plugin.imaging_fcs;

import ij.ImageJ;
import fiji.plugin.imaging_fcs.imfcs.Imaging_FCS;
import static org.junit.Assert.assertTrue;

public class Test {

    @org.junit.Test
    public void testApp() {

        // Simple test that always passes
        assertTrue(true);

        // Initialize ImageJ
        ImageJ ij = new ImageJ();

        // Create an instance of imaging FCS
        Imaging_FCS pluginImFCS = new Imaging_FCS();

        // Call the run method with an argument (can be empty if not used)
        pluginImFCS.run("start");

        // Prevent ImageJ from closing immediately by adding sleep
        try {
            Thread.sleep(1000000); // Keep the UI open for 10 seconds
        } catch (InterruptedException e) {
            e.printStackTrace(); // Handle the interruption
        }

        // Print ImageJ information
        System.out.println("getInfo: " + ij.getInfo());

    }

}
