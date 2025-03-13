#ifndef GPUFIT_ACF_NUMERICAL3D_CUH_INCLUDED
#define GPUFIT_ACF_NUMERICAL3D_CUH_INCLUDED

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// --- Helper Math Function Macros ---
#ifdef GPUFIT_DOUBLE
#    define my_exp(x) exp(x)
#    define my_erf(x) erf(x)
#    define my_fabs(x) fabs(x)
#else
#    define my_exp(x) __expf(x)
#    define my_erf(x) erff(x)
#    define my_fabs(x) fabsf(x)
#endif

// --- Constants ---
#define Z_STEPS 80
#define TOTAL_ITERATIONS (Z_STEPS * Z_STEPS)
#define CENTER_Z 40
#define DIVISOR_Z 20.0

// --- Helper Inline Functions ---

__device__ inline void computeComponent(REAL x, REAL a, REAL r, REAL sdt, REAL sp0t, REAL p0t, REAL SQRT_PI, REAL *part,
                                        REAL *deriv)
{
    REAL p1 = a + r;
    REAL p2 = a - r;
    REAL p3 = r;

    REAL p10 = p1 / sp0t;
    REAL p20 = p2 / sp0t;
    REAL p30 = p3 / sp0t;

    REAL exp1 = my_exp(-p10 * p10);
    REAL exp2 = my_exp(-p20 * p20);
    REAL exp3 = my_exp(-p30 * p30);
    REAL pexp = exp1 + exp2 - 2.0 * exp3;
    REAL perf = p1 * my_erf(p10) + p2 * my_erf(p20) - 2.0 * p3 * my_erf(p30);

    *part = (pexp * sp0t) / SQRT_PI + perf;

    REAL tsp0t = sp0t * sp0t * sp0t;
    REAL qp0t = p0t * p0t;

    REAL d1exp = 4 * x * (exp1 * p1 * p1);
    REAL d2exp = 4 * x * (exp2 * p2 * p2);
    REAL d3exp = 4 * x * (exp3 * p3 * p3);
    REAL d0exp = 2 * x * pexp;
    REAL dexp = d1exp + d2exp - 2.0 * d3exp;

    *deriv = (1.0 / sdt) * ((1.0 / SQRT_PI) * ((d0exp / sp0t) - (dexp / tsp0t)) + (dexp / SQRT_PI) * (sp0t / qp0t));
}

__device__ void calculate_acfnumerical3d(REAL const *parameters, int const n_fits, int const n_points, REAL *value,
                                         REAL *derivative, int const point_index, int const fit_index,
                                         int const chunk_index, char *user_info, std::size_t const user_info_size,
                                         int const num_v_coefs)
{
    // Extract lag time x from user_info
    REAL x = ((REAL *)user_info)[point_index];

    // --- Constants and Parameters ---
    const REAL PI = CUDART_PI;
    const REAL SQRT_PI = sqrt(PI);
    const REAL ridx = 1.3333;

    REAL N = parameters[0];
    REAL D = parameters[1];
    REAL ax = parameters[11];
    REAL ay = parameters[12];
    REAL s = parameters[13];
    REAL sz = parameters[14];
    REAL rx = parameters[15];
    REAL ry = parameters[16];
    REAL fTrip = parameters[9];
    REAL tTrip = parameters[10];
    REAL emLambda = parameters[20];
    REAL NA = parameters[21];

    // --- Precomputed Derived Values ---
    REAL srn = sqrt(ridx * ridx - NA * NA);
    REAL psfz = 2 * (emLambda / 1e9) * ridx / (NA * NA);
    REAL szeff = sqrt(1.0 / (pow(sz, -2.0) + pow(psfz, -2.0)));
    REAL sdt = sqrt(D * x);
    REAL dt1 = -0.5 * x / (sdt * sdt * sdt);
    REAL dz = (sz * sz) / 400.0;

    // --- Precompute z and psfz Arrays ---
    REAL z_values[Z_STEPS];
    REAL psfz_values[Z_STEPS];

    #pragma unroll
    for (int i = 0; i < Z_STEPS; ++i)
    {
        REAL z = (sz * (i - CENTER_Z)) / DIVISOR_Z;
        z_values[i] = z;
        psfz_values[i] = s + (NA * my_fabs(z)) / srn;
    }

    // --- Main Computation Loop ---
    REAL sum1 = 0.0;
    REAL sumd1 = 0.0;

    #pragma unroll 4
    for (int idx = 0; idx < TOTAL_ITERATIONS; ++idx)
    {
        int z1_index = idx / Z_STEPS;
        int z2_index = idx % Z_STEPS;

        REAL z1 = z_values[z1_index];
        REAL z2 = z_values[z2_index];
        REAL psfz1 = psfz_values[z1_index];
        REAL psfz2 = psfz_values[z2_index];

        REAL p0t = (8 * D * x + psfz1 * psfz1 + psfz2 * psfz2) / 2.0;
        REAL sp0t = sqrt(p0t);

        REAL xPart, xDeriv, yPart, yDeriv;
        computeComponent(x, ax, rx, sdt, sp0t, p0t, SQRT_PI, &xPart, &xDeriv);
        computeComponent(x, ay, ry, sdt, sp0t, p0t, SQRT_PI, &yPart, &yDeriv);

        REAL zdiff = z1 - z2;
        REAL z1exp = (2.0 / (sz * sz)) * (z1 * z1 + z2 * z2);
        REAL z2exp = (zdiff * zdiff) / (4 * D * x);
        REAL zExp = my_exp(-(z1exp + z2exp));
        REAL dt2 = 0.25 * (zdiff * zdiff) / (x * sdt * D * D);

        sum1 += (zExp * xPart * yPart * dz) / sdt;
        sumd1 += zExp * ((dt1 + dt2) * xPart * yPart + xDeriv * yPart + xPart * yDeriv) * dz;
    }

    // --- Final Calculations ---
    REAL volume3d = SQRT_PI * szeff * (4 * pow(ax * ay, 2));
    REAL acf1 = (sum1 * 1e6) / (4 * pow(ax * ay, 2) / (volume3d / (SQRT_PI * sz)));
    REAL Dpspim = (sumd1 * 1e6) / (4 * pow(ax * ay, 2) / (volume3d / (SQRT_PI * sz)));
    REAL triplet = 1.0 + fTrip / (1.0 - fTrip) * my_exp(-x / tTrip);

    value[point_index] = (1.0 / N)
            * (((1 - parameters[5]) * acf1 + pow(parameters[18], 2) * parameters[5] * 0.0)
               / (pow(1 - parameters[5] + parameters[18] * parameters[5], 2)))
            * triplet
        + parameters[4];

    // --- Derivatives ---
    derivative[0 * n_points + point_index] =
        (-1.0 / (N * N)) * ((1 - parameters[5]) * (acf1 / 1.6) + parameters[5] * 0.0) * triplet;
    derivative[1 * n_points + point_index] = (1.0 / N) * (Dpspim / 1.6);
    derivative[2 * n_points + point_index] = 0.0;
    derivative[3 * n_points + point_index] = 0.0;
    derivative[4 * n_points + point_index] = 1.0;
    derivative[5 * n_points + point_index] = 0.0;
    derivative[6 * n_points + point_index] = 0.0;
    derivative[7 * n_points + point_index] = 0.0;
    derivative[8 * n_points + point_index] = 0.0;
    derivative[9 * n_points + point_index] =
        (my_erf(-x / tTrip) * (1.0 / (1.0 - fTrip) + fTrip / ((1.0 - fTrip) * (1.0 - fTrip))))
        * (((acf1 / N) * triplet) + parameters[4]);
    derivative[10 * n_points + point_index] =
        (my_exp(-x / tTrip) * (fTrip * x) / ((1.0 - fTrip) * tTrip * tTrip)) * (((acf1 / N) * triplet) + parameters[4]);
}

#endif // GPUFIT_ACF_NUMERICAL3D_CUH_INCLUDED