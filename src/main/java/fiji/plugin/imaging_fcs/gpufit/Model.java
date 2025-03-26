package fiji.plugin.imaging_fcs.gpufit;

/**
 * Java binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
 * See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#java
 */
public enum Model {

    GAUSS_1D(0, 4),
    GAUSS_2D(1, 5),
    ACF_1D(2, 20),
    LINEAR_1D(3, 11),
    ACF_NUMERICAL_3D(4, 22);

    public final int id, numberParameters;

    Model(int id, int numberParameters) {
        this.id = id;
        this.numberParameters = numberParameters;
    }
}
