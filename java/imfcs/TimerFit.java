package imfcs;

public class TimerFit {

    long time1_increment;
    long time1_mem;

    public TimerFit() {
        time1_increment = 0;
    }

    public void tic() {
        time1_mem = System.currentTimeMillis();
    }

    public void toc() {
        long a1 = System.currentTimeMillis();
        time1_increment = time1_increment + (a1 - time1_mem);
        time1_mem = a1;
    }

    public long getTimeMillis() {
        return time1_increment;
    }

}
