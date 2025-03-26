#ifndef GPUFIT_MODELS_CUH_INCLUDED
#define GPUFIT_MODELS_CUH_INCLUDED

#include "acf_1d.cuh"
#include "acf_numerical3d.cuh"
#include "gauss_2d.cuh"
#include "linear_1d.cuh"

__device__ void calculate_model(ModelID const model_id, REAL const *parameters,
                                int const n_fits, int const n_points,
                                REAL *value, REAL *derivative,
                                int const point_index, int const fit_index,
                                int const chunk_index, char *user_info,
                                int const user_info_size,
                                int const num_v_coefs) // NEW
{
    switch (model_id)
    {
    case GAUSS_2D:
        calculate_gauss2d(parameters, n_fits, n_points, value, derivative,
                          point_index, fit_index, chunk_index, user_info,
                          user_info_size, num_v_coefs);
        break;
    case ACF_1D:
        calculate_acf1d(parameters, n_fits, n_points, value, derivative,
                        point_index, fit_index, chunk_index, user_info,
                        user_info_size, num_v_coefs);
        break;
    case LINEAR_1D:
        calculate_linear1d(parameters, n_fits, n_points, value, derivative,
                           point_index, fit_index, chunk_index, user_info,
                           user_info_size, num_v_coefs);
        break;
    case ACF_NUMERICAL_3D: // DAN
        calculate_acfnumerical3d(
            parameters, n_fits, n_points, value, derivative, point_index,
            fit_index, chunk_index, user_info, user_info_size, num_v_coefs);
        break;
    default:
        break;
    }
}

void configure_model(ModelID const model_id, int &n_parameters,
                     int &n_dimensions)
{
    switch (model_id)
    {
    case GAUSS_2D:
        n_parameters = 5;
        n_dimensions = 2;
        break;
    case ACF_1D:
        n_parameters = 20;
        n_dimensions = 1;
        break;
    case LINEAR_1D:
        n_parameters = 11;
        n_dimensions = 1;
        break;
    case ACF_NUMERICAL_3D:
        n_parameters = 22;
        n_dimensions = 1;
        break; // DAN
    default:
        break;
    }
}

#endif