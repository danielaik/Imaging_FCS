#ifndef GPUFIT_DEFINITIONS_H_INCLUDED
#define GPUFIT_DEFINITIONS_H_INCLUDED

// Precision
#ifdef GPUFIT_DOUBLE
#    define REAL double
#else
#    define REAL float
#endif // GPUFIT_DOUBLE

#include <stdexcept>

#ifdef USE_CUBLAS
#    include "cublas_v2.h"

#    ifdef GPUFIT_DOUBLE
#        define DECOMPOSE_LUP cublasDgetrfBatched
#        define SOLVE_LUP cublasDgetrsBatched
#    else
#        define DECOMPOSE_LUP cublasSgetrfBatched
#        define SOLVE_LUP cublasSgetrsBatched
#    endif

#    define SOLVE_EQUATION_SYSTEMS() solve_equation_systems_lup()
#else
#    define cublasHandle_t int
#    define SOLVE_EQUATION_SYSTEMS() solve_equation_systems_gj()
#endif

// Status
#define CUDA_CHECK_STATUS(cuda_function_call)                                  \
    if (cudaError_t const status = cuda_function_call)                         \
    {                                                                          \
        throw std::runtime_error(cudaGetErrorString(status));                  \
    }

#endif