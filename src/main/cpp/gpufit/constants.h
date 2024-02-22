#ifndef GPUFIT_CONSTANTS_H_INCLUDED
#define GPUFIT_CONSTANTS_H_INCLUDED

// fitting model ID

enum ModelID
{
    GAUSS_2D = 1,
    ACF_1D = 2,
    LINEAR_1D = 3,
    ACF_NUMERICAL_3D = 4 // DAN
};

// estimator ID
enum EstimatorID
{
    LSE = 0,
    MLE = 1
};

// fit state
enum FitState
{
    CONVERGED = 0,
    MAX_ITERATION = 1,
    SINGULAR_HESSIAN = 2,
    NEG_CURVATURE_MLE = 3,
    GPU_NOT_READY = 4
};

// return state
enum ReturnState
{
    OK = 0,
    ERROR = -1
};

enum DataLocation
{
    HOST = 0,
    DEVICE = 1
};

#endif