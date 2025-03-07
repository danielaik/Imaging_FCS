#ifndef GPUFIT_ACF1D_CUH_INCLUDED
#define GPUFIT_ACF1D_CUH_INCLUDED

__device__ void calculate_acf1d(REAL const *parameters, int const n_fits,
                                int const n_points, REAL *value,
                                REAL *derivative, int const point_index,
                                int const fit_index, int const chunk_index,
                                char *user_info,
                                std::size_t const user_info_size,
                                int const num_v_coefs) // NEW
{
    // indices

    REAL *user_info_float = (REAL *)user_info;
    REAL x = 0;
    x = user_info_float[point_index];
    double sqrpi = sqrt((double)3.14159265359);

    double p0t =
        sqrt(4 * (double)parameters[1] * x + pow((double)parameters[13], 2.0));
    double p1xt = (double)parameters[11] + (double)parameters[15]
        - (double)parameters[2] * x;
    double p2xt = (double)parameters[11] - (double)parameters[15]
        + (double)parameters[2] * x;
    double p3xt = (double)parameters[15] - (double)parameters[2] * x;
    double p4xt = 2 * pow((double)parameters[11], 2.0)
        + 3 * pow((double)parameters[15], 2.0)
        - 6 * x * (double)parameters[15] * (double)parameters[2]
        + 3 * pow(x * (double)parameters[2], 2.0);
    double p5xt = pow(p3xt, 2.0) + pow(p1xt, 2.0);
    double p6xt = pow(p3xt, 2.0) + pow(p2xt, 2.0);
    double p7xt = 2
        * (pow((double)parameters[11], 2.0) + pow((double)parameters[15], 2.0)
           - 2 * x * (double)parameters[15] * (double)parameters[2]
           + pow(x * (double)parameters[2], 2.0));
    double p1yt = (double)parameters[12] + (double)parameters[16]
        - (double)parameters[3] * x;
    double p2yt = (double)parameters[12] - (double)parameters[16]
        + (double)parameters[3] * x;
    double p3yt = (double)parameters[16] - (double)parameters[3] * x;
    double p4yt = 2 * pow((double)parameters[12], 2.0)
        + 3 * pow((double)parameters[16], 2.0)
        - 6 * x * (double)parameters[16] * (double)parameters[3]
        + 3 * pow(x * (double)parameters[3], 2.0);
    double p5yt = pow(p3yt, 2.0) + pow(p1yt, 2.0);
    double p6yt = pow(p3yt, 2.0) + pow(p2yt, 2.0);
    double p7yt = 2
        * (pow((double)parameters[12], 2.0) + pow((double)parameters[16], 2.0)
           - 2 * x * (double)parameters[16] * (double)parameters[3]
           + pow(x * (double)parameters[3], 2.0));

    double pexpxt = exp(-pow(p1xt / p0t, 2.0)) + exp(-pow(p2xt / p0t, 2.0))
        - 2 * exp(-pow(p3xt / p0t, 2.0));
    double perfxt = p1xt * erf(p1xt / p0t) + p2xt * erf(p2xt / p0t)
        - 2 * p3xt * erf(p3xt / p0t);
    double dDpexpxt = 2 * exp(-p4xt / pow(p0t, 2.0))
        * (exp(p5xt / pow(p0t, 2.0)) + exp(p6xt / pow(p0t, 2.0))
           - 2 * exp(p7xt / pow(p0t, 2.0)));
    double dvxperfxt =
        (erf(p2xt / p0t) + 2 * erf(p3xt / p0t) - erf(p1xt / p0t)) * x;
    double pexpyt = exp(-pow(p1yt / p0t, 2.0)) + exp(-pow(p2yt / p0t, 2.0))
        - 2 * exp(-pow(p3yt / p0t, 2.0));
    double dDpexpyt = 2 * exp(-p4yt / pow(p0t, 2.0))
        * (exp(p5yt / pow(p0t, 2.0)) + exp(p6yt / pow(p0t, 2.0))
           - 2 * exp(p7yt / pow(p0t, 2.0)));
    double dvyperfyt =
        (erf(p2yt / p0t) + 2 * erf(p3yt / p0t) - erf(p1yt / p0t)) * x;
    double perfyt = p1yt * erf(p1yt / p0t) + p2yt * erf(p2yt / p0t)
        - 2 * p3yt * erf(p3yt / p0t);
    double pplane1 = (p0t / sqrpi * pexpxt + perfxt)
        * (p0t / sqrpi * pexpyt + perfyt)
        / (4 * (double)parameters[11] * (double)parameters[12])
        * ((double)parameters[17]
           / ((double)parameters[11] * (double)parameters[12]));

    double p0t2 =
        sqrt(4 * (double)parameters[6] * x + pow((double)parameters[13], 2));
    double p1xt2 = (double)parameters[11] + (double)parameters[15]
        - (double)parameters[2] * x;
    double p2xt2 = (double)parameters[11] - (double)parameters[15]
        + (double)parameters[2] * x;
    double p3xt2 = (double)parameters[15] - (double)parameters[2] * x;
    double p4xt2 = 2 * pow((double)parameters[11], 2)
        + 3 * pow((double)parameters[15], 2)
        - 6 * x * (double)parameters[15] * (double)parameters[2]
        + 3 * pow(x * (double)parameters[2], 2);
    double p5xt2 = pow(p3xt2, 2) + pow(p1xt2, 2);
    double p6xt2 = pow(p3xt2, 2) + pow(p2xt2, 2);
    double p7xt2 = 2
        * (pow((double)parameters[11], 2) + pow((double)parameters[15], 2)
           - 2 * x * (double)parameters[15] * (double)parameters[2]
           + pow(x * (double)parameters[2], 2));
    double p1yt2 = (double)parameters[12] + (double)parameters[16]
        - (double)parameters[3] * x;
    double p2yt2 = (double)parameters[12] - (double)parameters[16]
        + (double)parameters[3] * x;
    double p3yt2 = (double)parameters[16] - (double)parameters[3] * x;
    double p4yt2 = 2 * pow((double)parameters[12], 2)
        + 3 * pow((double)parameters[16], 2)
        - 6 * x * (double)parameters[16] * (double)parameters[3]
        + 3 * pow(x * (double)parameters[3], 2);
    double p5yt2 = pow(p3yt2, 2) + pow(p1yt2, 2);
    double p6yt2 = pow(p3yt2, 2) + pow(p2yt2, 2);
    double p7yt2 = 2
        * (pow((double)parameters[12], 2) + pow((double)parameters[16], 2)
           - 2 * x * (double)parameters[16] * (double)parameters[3]
           + pow(x * (double)parameters[3], 2));
    double pexpxt2 = exp(-pow(p1xt2 / p0t2, 2)) + exp(-pow(p2xt2 / p0t2, 2))
        - 2 * exp(-pow(p3xt2 / p0t2, 2));
    double perfxt2 = p1xt2 * erf(p1xt2 / p0t2) + p2xt2 * erf(p2xt2 / p0t2)
        - 2 * p3xt2 * erf(p3xt2 / p0t2);
    double dDpexpxt2 = 2 * exp(-p4xt2 / pow(p0t2, 2))
        * (exp(p5xt2 / pow(p0t2, 2)) + exp(p6xt2 / pow(p0t2, 2))
           - 2 * exp(p7xt2 / pow(p0t2, 2)));
    double dvxperfxt2 =
        (erf(p2xt2 / p0t2) + 2 * erf(p3xt2 / p0t2) - erf(p1xt2 / p0t2)) * x;
    double pexpyt2 = exp(-pow(p1yt2 / p0t2, 2)) + exp(-pow(p2yt2 / p0t2, 2))
        - 2 * exp(-pow(p3yt2 / p0t2, 2));
    double dDpexpyt2 = 2 * exp(-p4yt2 / pow(p0t2, 2))
        * (exp(p5yt2 / pow(p0t2, 2)) + exp(p6yt2 / powf(p0t2, 2))
           - 2 * exp(p7yt2 / powf(p0t2, 2)));
    double dvyperfyt2 =
        (erf(p2yt2 / p0t2) + 2 * erf(p3yt2 / p0t2) - erf(p1yt2 / p0t2)) * x;
    double perfyt2 = p1yt2 * erf(p1yt2 / p0t2) + p2yt2 * erf(p2yt2 / p0t2)
        - 2 * p3yt2 * erf(p3yt2 / p0t2);
    double pplane2 = (p0t2 / sqrpi * pexpxt2 + perfxt2)
        * (p0t2 / sqrpi * pexpyt2 + perfyt2)
        / (4 * pow((double)parameters[11] * (double)parameters[12], 2)
           / (double)parameters[17]);

    double triplet = 1
        + (double)parameters[9] / (1 - (double)parameters[9])
            * exp(-x / (double)parameters[10]);

    value[point_index] = (1 / (double)parameters[0])
            * ((1 - (double)parameters[5]) * pplane1
               + powf((double)parameters[18], 2) * (double)parameters[5] * pplane2)
            / pow(1 - (double)parameters[5]
                      + (double)parameters[18] * (double)parameters[5],
                  2)
            * triplet
        + (double)parameters[4];

    double dDplat = (1 / (sqrpi * p0t))
        * (dDpexpyt * x * (p0t / sqrpi * pexpxt + perfxt)
           + dDpexpxt * x * (p0t / sqrpi * pexpyt + perfyt))
        / (4 * powf((double)parameters[11] * (double)parameters[12], 2.0)
           / (double)parameters[17]);

    double dDplat2 = (1 / (sqrpi * p0t2))
        * (dDpexpyt2 * x * (p0t2 / sqrpi * pexpxt2 + perfxt2)
           + dDpexpxt2 * x * (p0t2 / sqrpi * pexpyt2 + perfyt2))
        / (4 * pow((double)parameters[11] * (double)parameters[12], 2)
           / (double)parameters[17]);

    double dtripletFtrip = exp(-x / (double)parameters[10])
        * (1 / (1 - (double)parameters[9])
           + (double)parameters[9] / pow(1 - (double)parameters[9], 2));
    double dtripletTtrip = exp(-x / (double)parameters[10])
        * ((double)parameters[9] * x)
        / ((1 - (double)parameters[9]) * pow((double)parameters[10], 2));

    double pf1 = (1 - (double)parameters[5])
        / (1 - (double)parameters[5]
           + (double)parameters[18] * (double)parameters[5]);
    double pf2 = (pow((double)parameters[18], 2) * (double)parameters[5])
        / (1 - (double)parameters[5]
           + (double)parameters[18] * (double)parameters[5]);
    double dfnom = pow(1 - (double)parameters[5]
                           + (double)parameters[18] * (double)parameters[5],
                       3);
    double df21 = 1 - (double)parameters[5]
        + (double)parameters[18] * (double)parameters[5]
        - 2 * (double)parameters[18];
    double df22 = pow((double)parameters[18], 2)
        * (1 + (double)parameters[5]
           - (double)parameters[18] * (double)parameters[5]);

    double pacf = (1 / (double)parameters[0])
            * ((1 - (double)parameters[5]) * pplane1
               + powf((double)parameters[18], 2) * (double)parameters[5] * pplane2)
            / pow(1 - (double)parameters[5]
                      + (double)parameters[18] * (double)parameters[5],
                  2)
            * triplet
        + (double)parameters[4];

    REAL *current_derivatives = derivative + point_index;

    current_derivatives[0 * n_points] =
        (float)(-1 / pow((double)parameters[0], 2)) * (pf1 * pplane1 + pf2 * pplane2)
        * triplet;
    current_derivatives[1 * n_points] = (1 / parameters[0])
        * (float)(pf1 * dDplat);
    current_derivatives[2 * n_points] = (1 / parameters[0])
        * (float)((pf1 * ((p0t / sqrpi * pexpyt + perfyt) * dvxperfxt)
                       / (4
                          * pow((double)parameters[11] * (double)parameters[12],
                                2)
                          / parameters[17])
                   + pf2 * ((p0t2 / sqrpi * pexpyt2 + perfyt2) * dvxperfxt2)
                       / (4
                          * pow((double)parameters[11] * (double)parameters[12],
                                2)
                          / parameters[17]))
                  * triplet);
    current_derivatives[3 * n_points] = (1 / parameters[0])
        * (float)((pf1 * ((p0t / sqrpi * pexpxt + perfxt) * dvyperfyt)
                       / (4
                          * pow((double)parameters[11] * (double)parameters[12],
                                2)
                          / parameters[17])
                   + pf2 * ((p0t2 / sqrpi * pexpxt2 + perfxt2) * dvyperfyt2)
                       / (4
                          * pow((double)parameters[11] * (double)parameters[12],
                                2)
                          / parameters[17]))
                  * triplet);
    current_derivatives[4 * n_points] = 1.0;
    current_derivatives[5 * n_points] = (1 / parameters[0])
        * (float)((1 / dfnom) * (df21 * pplane1 + df22 * pplane2) * triplet);
    current_derivatives[6 * n_points] = (1 / parameters[0])
        * (float)(pf2 * dDplat2 * triplet);
    current_derivatives[9 * n_points] = (float)dtripletFtrip * pacf;
    current_derivatives[10 * n_points] = (float)dtripletTtrip * pacf;
    // current_derivatives[11 * n_points] = 0.0;
    // current_derivatives[12 * n_points] = 0.0;
    // current_derivatives[13 * n_points] = 0.0;
    // current_derivatives[14 * n_points] = 0.0;
    // current_derivatives[15 * n_points] = 0.0;
    // current_derivatives[16 * n_points] = 0.0;
    // current_derivatives[17 * n_points] = 0.0;
    // current_derivatives[18 * n_points] = 0.0;
    // current_derivatives[19 * n_points] = 0.0;
}

#endif