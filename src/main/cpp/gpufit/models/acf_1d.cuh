#ifndef GPUFIT_ACF1D_CUH_INCLUDED
#define GPUFIT_ACF1D_CUH_INCLUDED

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

// --- Helper Math Function Macros ---
#ifdef GPUFIT_DOUBLE
#    define my_exp(x) exp(x)
#    define my_pow(x, y) pow(x, y)
#    define my_erf(x) erf(x)
#    define my_sqrt(x) sqrt(x)
#else
#    define my_exp(x) __expf(x)
#    define my_pow(x, y) __powf(x, y)
#    define my_erf(x) erff(x)
#    define my_sqrt(x) sqrtf(x)
#endif

// --- Helper Structures ---
struct PerfTerms
{
    REAL exp; // Exponential term
    REAL perf; // Error function term
    REAL dExp; // Derivative w.r.t. diffusion
    REAL dPerf; // Derivative w.r.t. velocity
};

// --- Helper Functions ---

__device__ inline PerfTerms calculatePerfTerms(REAL x, REAL v, REAL a, REAL r, REAL p0t, REAL SQRT_PI)
{
    REAL term1 = a + r - v * x;
    REAL term2 = a - r + v * x;
    REAL term3 = r - v * x;
    REAL term4 = 2 * my_pow(a, 2) + 3 * my_pow(r, 2) - 6 * x * r * v + 3 * my_pow(x * v, 2);
    REAL term5 = my_pow(term3, 2) + my_pow(term1, 2);
    REAL term6 = my_pow(term3, 2) + my_pow(term2, 2);
    REAL term7 = 2 * (my_pow(a, 2) + my_pow(r, 2) - 2 * x * r * v + my_pow(x * v, 2));

    REAL expTerm =
        my_exp(-my_pow(term1 / p0t, 2)) + my_exp(-my_pow(term2 / p0t, 2)) - 2 * my_exp(-my_pow(term3 / p0t, 2));
    REAL erfTerm = term1 * my_erf(term1 / p0t) + term2 * my_erf(term2 / p0t) - 2 * term3 * my_erf(term3 / p0t);
    REAL dExpTerm = 2 * my_exp(-term4 / my_pow(p0t, 2))
        * (my_exp(term5 / my_pow(p0t, 2)) + my_exp(term6 / my_pow(p0t, 2)) - 2 * my_exp(term7 / my_pow(p0t, 2)));
    REAL dPerf = (my_erf(term2 / p0t) + 2 * my_erf(term3 / p0t) - my_erf(term1 / p0t)) * x;

    PerfTerms terms = { expTerm, erfTerm, dExpTerm, dPerf };
    return terms;
}

__device__ inline REAL calculatePlat(PerfTerms perfX, PerfTerms perfY, REAL p0t, REAL SQRT_PI, REAL ax, REAL ay,
                                     REAL fitObservationVolume)
{
    return (p0t / SQRT_PI * perfX.exp + perfX.perf) * (p0t / SQRT_PI * perfY.exp + perfY.perf)
        / (4 * my_pow(ax * ay, 2) / fitObservationVolume);
}

__device__ inline REAL calculateDPlat(PerfTerms perfX, PerfTerms perfY, REAL p0t, REAL SQRT_PI, REAL x, REAL ax,
                                      REAL ay, REAL fitObservationVolume)
{
    return (1 / (SQRT_PI * p0t))
        * (perfY.dExp * x * (p0t / SQRT_PI * perfX.exp + perfX.perf)
           + perfX.dExp * x * (p0t / SQRT_PI * perfY.exp + perfY.perf))
        / (4 * my_pow(ax * ay, 2) / fitObservationVolume);
}

// --- Main Device Function ---
__device__ void calculate_acf1d(REAL const *parameters, int const n_fits, int const n_points, REAL *value,
                                REAL *derivative, int const point_index, int const fit_index, int const chunk_index,
                                char *user_info, std::size_t const user_info_size, int const num_v_coefs)
{
    // Extract x from user_info
    REAL *user_info_float = (REAL *)user_info;
    REAL x = user_info_float[point_index];

    // Constants
    const REAL SQRT_PI = my_sqrt(CUDART_PI);

    // Extract parameters (assuming REAL is float for GPU performance)
    REAL N = parameters[0]; // Number of particles
    REAL D = parameters[1]; // Diffusion coefficient 1
    REAL vx = parameters[2]; // Velocity in x
    REAL vy = parameters[3]; // Velocity in y
    REAL G = parameters[4]; // Offset
    REAL F2 = parameters[5]; // Fraction of second component
    REAL D2 = parameters[6]; // Diffusion coefficient 2
    REAL fTrip = parameters[9]; // Triplet fraction
    REAL tTrip = parameters[10]; // Triplet time
    REAL ax = parameters[11]; // Axial size x
    REAL ay = parameters[12]; // Axial size y
    REAL s = parameters[13]; // PSF size
    REAL rx = parameters[15]; // Radial size x
    REAL ry = parameters[16]; // Radial size y
    REAL fitObservationVolume = parameters[17]; // Observation volume
    REAL q2 = parameters[18]; // Relative amplitude of second component

    // Precompute p0t for each component
    REAL p0t1 = my_sqrt(4 * D * x + my_pow(s, 2));
    REAL p0t2 = my_sqrt(4 * D2 * x + my_pow(s, 2));

    // Compute perf terms for each component
    PerfTerms perfX1 = calculatePerfTerms(x, vx, ax, rx, p0t1, SQRT_PI);
    PerfTerms perfY1 = calculatePerfTerms(x, vy, ay, ry, p0t1, SQRT_PI);
    PerfTerms perfX2 = calculatePerfTerms(x, vx, ax, rx, p0t2, SQRT_PI);
    PerfTerms perfY2 = calculatePerfTerms(x, vy, ay, ry, p0t2, SQRT_PI);

    // Compute plat and dDplat for each component
    REAL plat1 = calculatePlat(perfX1, perfY1, p0t1, SQRT_PI, ax, ay, fitObservationVolume);
    REAL dDplat1 = calculateDPlat(perfX1, perfY1, p0t1, SQRT_PI, x, ax, ay, fitObservationVolume);
    REAL plat2 = calculatePlat(perfX2, perfY2, p0t2, SQRT_PI, ax, ay, fitObservationVolume);
    REAL dDplat2 = calculateDPlat(perfX2, perfY2, p0t2, SQRT_PI, x, ax, ay, fitObservationVolume);

    // Triplet correction
    REAL triplet = 1 + fTrip / (1 - fTrip) * my_exp(-x / tTrip);
    REAL dTripletFtrip = my_exp(-x / tTrip) * (1 / (1 - fTrip) + fTrip / my_pow(1 - fTrip, 2));
    REAL dTripletTtrip = my_exp(-x / tTrip) * (fTrip * x) / ((1 - fTrip) * my_pow(tTrip, 2));

    // Correction factors
    REAL denominator = 1 - F2 + q2 * F2;
    REAL pf1 = (1 - F2) / denominator;
    REAL pf2 = (my_pow(q2, 2) * F2) / denominator;

    // Derivative factors for F2
    REAL dfNom = my_pow(denominator, 3);
    REAL df21 = 1 - F2 + q2 * F2 - 2 * q2;
    REAL df22 = my_pow(q2, 2) * (1 + F2 - q2 * F2);

    // Value computation
    REAL weightedAcf = pf1 * plat1 + pf2 * plat2;
    value[point_index] = (1 / N) * weightedAcf * triplet + G;

    // Derivatives
    REAL *current_derivatives = derivative + point_index;
    REAL pacf = (1 / N) * weightedAcf * triplet + G;

    current_derivatives[0 * n_points] = (-1 / (N * N)) * weightedAcf * triplet; // d/dN
    current_derivatives[1 * n_points] = (1 / N) * pf1 * dDplat1 * triplet; // d/dD
    current_derivatives[2 * n_points] = (1 / N) * // d/dvx
        (pf1 * perfX1.dPerf * plat1 + pf2 * perfX2.dPerf * plat2) * triplet;
    current_derivatives[3 * n_points] = (1 / N) * // d/dvy
        (pf1 * perfY1.dPerf * plat1 + pf2 * perfY2.dPerf * plat2) * triplet;
    current_derivatives[4 * n_points] = 1.0; // d/dG
    current_derivatives[5 * n_points] = (1 / N) * (1 / dfNom) * // d/dF2
        (df21 * plat1 + df22 * plat2) * triplet;
    current_derivatives[6 * n_points] = (1 / N) * pf2 * dDplat2 * triplet; // d/dD2
    current_derivatives[9 * n_points] = dTripletFtrip * pacf; // d/dfTrip
    current_derivatives[10 * n_points] = dTripletTtrip * pacf; // d/dtTrip
}

#endif