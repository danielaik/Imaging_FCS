#ifndef GPU_FIT_H_INCLUDED
#define GPU_FIT_H_INCLUDED

/* --------------------------------------------------------------------------------------------------------
NOTES:

This gpufit code is consolidated from an open-source code. Please see
https://gpufit.readthedocs.io/en/latest/ for more information.
We added ACF and bleach correction fitting functions. Furthermore,
we have also placed the CUDA ACF calculations codes here.

Some implementation details:
1. void calcacf3 kernel function calculates the block transformation values of
the intensity.
2. void calcacf2a kernel function calculates the arrays according to different
time bins in different parts of the correlation function.
3. void calcacf2b kernel function calculates the value of the auto or
cross-correlation at every lag time. This function also performs the G1 analysis
in N and B calculation.
4. void calc_data_bleach_correction kernel function is an averaging step in
temporal dimension for every ave number of points, prior to performing bleach
correction fitting.
5. void calc_binning kernel function performs binning of spatial data.
6. void bleachcorrection kernel function performs polynomial bleach correction
given polynomial order and coefficients. It is done prior to calcacf3, calcacf2a
and calcacf2b.
7. Kindly also take a look at the CPU functions for a detailed description of
the variables.
8. argument float* data is the intensity array input on which the N and B or the
autocorrelation or the cross-correlation has to be calculates.
9. argument double* data1 is the output array where the values of auto and
cross-correlation are calculated.
---------------------------------------------------------------------------------------------------------
*/

// declare as a flag during compilation, ie. -D USE_CUBLAS
// Without USE_CUBLAS, we are using solve_equation_systems_gj(), i.e.
// cuda_gaussjordan.cu
// #define USE_CUBLAS

// from gpufit.h
#ifdef __linux__
#    define VISIBLE __attribute__((visibility("default")))
#else
#    define VISIBLE
#endif

#include <cstddef>
#include <stdexcept>

#include "constants.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C"
{
#endif

    VISIBLE int gpufit(std::size_t n_fits, std::size_t n_points, REAL *data,
                       REAL *weights, int model_id, REAL *initial_parameters,
                       REAL tolerance, int max_n_iterations,
                       int num_v_coefs, // NEW
                       int *parameters_to_fit, int estimator_id,
                       std::size_t user_info_size, char *user_info,
                       REAL *output_parameters, int *output_states,
                       REAL *output_chi_squares, int *output_n_iterations);

    VISIBLE int gpufit_constrained(std::size_t n_fits, std::size_t n_points,
                                   REAL *data, REAL *weights, int model_id,
                                   REAL *initial_parameters, REAL *constraints,
                                   int *constraint_types, REAL tolerance,
                                   int max_n_iterations, int *parameters_to_fit,
                                   int estimator_id, std::size_t user_info_size,
                                   char *user_info, REAL *output_parameters,
                                   int *output_states, REAL *output_chi_squares,
                                   int *output_n_iterations);

    VISIBLE int gpufit_cuda_interface(
        std::size_t n_fits, std::size_t n_points, REAL *gpu_data,
        REAL *gpu_weights, int model_id, REAL tolerance, int max_n_iterations,
        int *parameters_to_fit, int estimator_id, std::size_t user_info_size,
        char *gpu_user_info, REAL *gpu_fit_parameters, int *gpu_output_states,
        REAL *gpu_output_chi_squares, int *gpu_output_n_iterations);

    VISIBLE int gpufit_constrained_cuda_interface(
        std::size_t n_fits, std::size_t n_points, REAL *gpu_data,
        REAL *gpu_weights, int model_id, REAL tolerance, int max_n_iterations,
        int *parameters_to_fit, REAL *gpu_constraints, int *constraint_types,
        int estimator_id, std::size_t user_info_size, char *gpu_user_info,
        REAL *gpu_fit_parameters, int *gpu_output_states,
        REAL *gpu_output_chi_squares, int *gpu_output_n_iterations);

    VISIBLE char const *gpufit_get_last_error();

    // returns 1 if cuda is available and 0 otherwise
    VISIBLE int gpufit_cuda_available();

    VISIBLE int gpufit_get_cuda_version(int *runtime_version,
                                        int *driver_version);

    VISIBLE int gpufit_portable_interface(int argc, void *argv[]);

    VISIBLE int gpufit_constrained_portable_interface(int argc, void *argv[]);

#ifdef __cplusplus
}
#endif

#endif