#ifndef GPUFIT_ACF_NUMERICAL3D_CUH_INCLUDED
#define GPUFIT_ACF_NUMERICAL3D_CUH_INCLUDED

// DAN
__device__ void calculate_acfnumerical3d(
    REAL const *parameters, int const n_fits, int const n_points, REAL *value,
    REAL *derivative, int const point_index, int const fit_index,
    int const chunk_index, char *user_info, std::size_t const user_info_size,
    int const num_v_coefs) // NEW
{
    // indices

    REAL *user_info_float = (REAL *)user_info;
    REAL x = 0;
    x = user_info_float[point_index];

    double pi = 3.14159265359;
    double sqrpi = sqrt(pi);
    double const1 = sqrt(1 / pi);
    double ridx = 1.3333; // refractive index of mounting medium, here water
    double ax = (double)parameters[11];
    double ay = (double)parameters[12];
    double s = (double)parameters[13];
    double sz = (double)parameters[14];

    double emLambda = (double) parameters[20];
    double NA = (double) parameters[21];

    double psfz = 2 * (emLambda / pow(10.0, 9.0)) * ridx / pow(NA, 2.0);
    double szeff = sqrt(1.0 / (pow(sz, -2.0) + pow(psfz, -2.0)));

    double rx = (double)parameters[15];
    double ry = (double)parameters[16];

    double srn = sqrt(pow(ridx, 2.0) - pow(NA, 2.0));
    double z1;
    double z2;

    double p1xt = ax + rx;
    double p2xt = ax - rx;
    double p3xt = rx;
    double p1yt = ay + ry;
    double p2yt = ay - ry;
    double p3yt = ry;

    double sum1;
    double sumd1;

    int i;

    double p00 = s;
    double p1x0 = ax;
    double p1y0 = ay;
    double pexpx0 = 2 * exp(-pow(p1x0 / p00, 2)) - 2;
    double perfx0 = 2 * p1x0 * erf(p1x0 / p00);
    double pexpy0 = 2 * exp(-pow(p1y0 / p00, 2)) - 2;
    double perfy0 = 2 * p1y0 * erf(p1y0 / p00);

    double volume3d = sqrpi * szeff * 4 * pow(ax * ay, 2)
        / ((p00 / sqrpi * pexpx0 + perfx0) * (p00 / sqrpi * pexpy0 + perfy0));

    double sdt = sqrt((double)parameters[1] * x);
    //	double adt = ((double) parameters[1]*x);

    sum1 = 0.0;
    sumd1 = 0.0;
    for (i = 0; i < 6401; ++i)
    {
        int counter = i;
        int outerloop = counter / 80;
        int z1calculator = outerloop - 40;
        z1 = (sz * z1calculator) / 20;
        int z2calculator = (i % 80) - 40;
        double psfxz1 = s + (NA * abs(z1)) / srn;
        z2 = (sz * z2calculator) / 20;
        double psfxz2 = s + (NA * abs(z2)) / srn;

        double p0t =
            ((8 * (double)parameters[1] * x) + pow(psfxz1, 2) + pow(psfxz2, 2))
            / 2;
        double sp0t = sqrt(p0t);
        double tsp0t = pow(sp0t, 3);
        double qp0t = pow(p0t, 2);

        double p10xt = p1xt / sp0t;
        double p20xt = p2xt / sp0t;
        double p30xt = p3xt / sp0t;
        double p1expxt = exp(-pow(p10xt, 2));
        double p2expxt = exp(-pow(p20xt, 2));
        double p3expxt = exp(-pow(p30xt, 2));
        double pexpxt = p1expxt + p2expxt - (2 * p3expxt);
        double perfxt =
            (p1xt * erf(p10xt)) + (p2xt * erf(p20xt)) - (2 * p3xt * erf(p30xt));
        double d1expx = 4 * x * (p1expxt * pow(p1xt, 2));
        double d2expx = 4 * x * (p2expxt * pow(p2xt, 2));
        double d3expx = 4 * x * (p3expxt * pow(p3xt, 2));
        double d0expx = 2 * x * (pexpxt);
        double dexpx = d1expx + d2expx - (2 * d3expx);
        double xpart = ((pexpxt * sp0t) / sqrpi) + perfxt;
        double xder = (1 / sdt)
            * (const1 * ((d0expx / sp0t) - (dexpx / tsp0t))
               + (dexpx / sqrpi) * (sp0t / qp0t));

        double p10yt = p1yt / sp0t;
        double p20yt = p2yt / sp0t;
        double p30yt = p3yt / sp0t;
        double p1expyt = exp(-pow(p10yt, 2));
        double p2expyt = exp(-pow(p20yt, 2));
        double p3expyt = exp(-pow(p30yt, 2));
        double pexpyt = p1expyt + p2expyt - (2 * p3expyt);
        double perfyt =
            (p1yt * erf(p10yt)) + (p2yt * erf(p20yt)) - (2 * p3yt * erf(p30yt));
        double d1expy = 4 * x * (p1expyt * pow(p1yt, 2));
        double d2expy = 4 * x * (p2expyt * pow(p2yt, 2));
        double d3expy = 4 * x * (p3expyt * pow(p3yt, 2));
        double d0expy = 2 * x * (pexpyt);
        double dexpy = d1expy + d2expy - (2 * d3expy);
        double ypart = ((pexpyt * sp0t) / sqrpi) + perfyt;
        double yder = (1 / sdt)
            * (const1 * ((d0expy / sp0t) - (dexpy / tsp0t))
               + (dexpy / sqrpi) * (sp0t / qp0t));

        double zdiff = (z1 - z2);
        double z1exp = (2 / pow(sz, 2)) * (pow(z1, 2) + pow(z2, 2));
        double z2exp = (pow(zdiff, 2)) / (4 * (double)parameters[1] * x);
        double zexp = exp(-(z1exp + z2exp));

        double dt1 = -(0.5 * x) / (pow(sdt, 3));
        double dt2 = (0.25 * (pow((zdiff), 2)))
            / (x * sdt * pow((double)parameters[1], 2));

        // ASHWIN:check that the two are correct
        sum1 += ((zexp * xpart * ypart) * ((sz * sz) / 400)) / sdt;
        sumd1 += zexp
            * ((dt1 + dt2) * xpart * ypart + xpart * yder + ypart * xder)
            * (sz * sz / 400);
    }

    /*

        double sqrpi = sqrt((double) 3.14159265359);
        double p0t = sqrt(4 * (double) parameters[1] * x + pow((double)
     parameters[13], 2.0)); double p1xt = (double) parameters[11] + (double)
     parameters[15] - (double) parameters[2] * x; double p2xt = (double)
     parameters[11] - (double) parameters[15] + (double) parameters[2] * x;
        double p3xt = (double) parameters[15] - (double) parameters[2] * x;
        double p4xt = 2 * pow((double) parameters[11], 2.0) + 3 * pow((double)
     parameters[15], 2.0) - 6 * x * (double) parameters[15] * (double)
     parameters[2] + 3 * pow(x * (double) parameters[2], 2.0); double p5xt =
     pow(p3xt, 2.0) + pow(p1xt, 2.0); double p6xt = pow(p3xt, 2.0) +
     pow(p2xt, 2.0); double p7xt = 2 * (pow((double) parameters[11], 2.0) +
     pow((double) parameters[15], 2.0) - 2 * x * (double) parameters[15] *
     (double) parameters[2] + pow(x * (double) parameters[2], 2.0)); double p1yt
     = (double) parameters[12] + (double) parameters[16] - (double)
     parameters[3] * x; double p2yt = (double) parameters[12] - (double)
     parameters[16] + (double) parameters[3] * x; double p3yt = (double)
     parameters[16] - (double) parameters[3] * x; double p4yt = 2 * pow((double)
     parameters[12], 2.0) + 3 * pow((double) parameters[16], 2.0) - 6 * x *
     (double) parameters[16] * (double) parameters[3] + 3 * pow(x * (double)
     parameters[3], 2.0); double p5yt = pow(p3yt, 2.0) + pow(p1yt, 2.0); double
     p6yt = pow(p3yt, 2.0) + pow(p2yt, 2.0); double p7yt = 2 * (pow((double)
     parameters[12], 2.0) + pow((double) parameters[16], 2.0) - 2 * x * (double)
     parameters[16] * (double) parameters[3] + pow(x * (double)
     parameters[3], 2.0));

        double pexpxt = exp(-pow(p1xt / p0t, 2.0)) + exp(-pow(p2xt / p0t, 2.0))
     - 2 * exp(-pow(p3xt / p0t, 2.0)); double perfxt = p1xt * erf(p1xt / p0t) +
     p2xt * erf(p2xt / p0t) - 2 * p3xt * erf(p3xt / p0t); double dDpexpxt = 2 *
     exp(-p4xt / pow(p0t, 2.0)) * (exp(p5xt / pow(p0t, 2.0)) + exp(p6xt /
     pow(p0t, 2.0)) - 2 * exp(p7xt / pow(p0t, 2.0))); double dvxperfxt =
     (erf(p2xt / p0t) + 2 * erf(p3xt / p0t) - erf(p1xt / p0t)) * x; double
     pexpyt = exp(-pow(p1yt / p0t, 2.0)) + exp(-pow(p2yt / p0t, 2.0)) - 2 *
     exp(-pow(p3yt / p0t, 2.0)); double dDpexpyt = 2 * exp(-p4yt /
     pow(p0t, 2.0)) * (exp(p5yt / pow(p0t, 2.0)) + exp(p6yt / pow(p0t, 2.0)) - 2
     * exp(p7yt / pow(p0t, 2.0))); double dvyperfyt = (erf(p2yt / p0t) + 2 *
     erf(p3yt / p0t) - erf(p1yt / p0t)) * x; double perfyt = p1yt * erf(p1yt /
     p0t) + p2yt * erf(p2yt / p0t) - 2 * p3yt * erf(p3yt / p0t); double pplane1
     = (p0t / sqrpi * pexpxt + perfxt) * (p0t / sqrpi * pexpyt + perfyt) / (4 *
     (double) parameters[11]*(double) parameters[12]) * ((double) parameters[17]
     / ((double) parameters[11]*(double) parameters[12])); double pspim1 = 1 /
     sqrt(1 + (4 * (double) parameters[1] * x) / powf((double) parameters[14],
     2));
      //  double acf1 = pplane1 * pspim1;
     //   double acf1=sum1;
     */
    double acf1 =
        (sum1 * 1000000) / (4 * pow(ax * ay, 2) / (volume3d / (sqrpi * sz)));
    double Dpspim =
        (sumd1 * 1000000) / (4 * pow(ax * ay, 2) / (volume3d / (sqrpi * sz)));

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
    double pspim2 = 1
        / sqrt(1
               + (4 * (double)parameters[6] * x)
                   / pow((double)parameters[14], 2));
    double acf2 = pplane2 * pspim2;

    double triplet = 1
        + (double)parameters[9] / (1 - (double)parameters[9])
            * exp(-x / (double)parameters[10]);

    value[point_index] = (1 / (double)parameters[0])
            * ((1 - (double)parameters[5]) * acf1
               + powf((double)parameters[18], 2) * (double)parameters[5] * acf2)
            / pow(1 - (double)parameters[5]
                      + (double)parameters[18] * (double)parameters[5],
                  2)
            * triplet
        + (double)parameters[4];

    //  double dDplat = (1 / (sqrpi * p0t)) * (dDpexpyt * x * (p0t / sqrpi *
    //  pexpxt + perfxt) + dDpexpxt * x * (p0t / sqrpi * pexpyt + perfyt)) / (4
    //  * powf((double) parameters[11]*(double) parameters[12], 2.0) / (double)
    //  parameters[17]);
    //   double dDpspim = -4 * x / (2 * pow((double) parameters[14], 2) *
    //   pow(sqrt(1 + (4 * (double) parameters[1] * x) / pow((double)
    //   parameters[14], 2)), 3));

    //  double dDplat2 = (1 / (sqrpi * p0t2)) * (dDpexpyt2 * x * (p0t2 / sqrpi *
    //  pexpxt2 + perfxt2) + dDpexpxt2 * x * (p0t2 / sqrpi * pexpyt2 + perfyt2))
    //  / (4 * pow((double) parameters[11]*(double) parameters[12], 2) /
    //  (double) parameters[17]); double dDpspim2 = -4 * x / (2 * pow((double)
    //  parameters[14], 2) * pow(sqrt(1 + (4 * (double) parameters[6] * x) /
    //  pow((double) parameters[14], 2)), 3));

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
    //  double dfnom = pow(1 - (double) parameters[5] + (double) parameters[18]
    //  * (double) parameters[5], 3);
    //   double df21 = 1 - (double) parameters[5] + (double) parameters[18] *
    //   (double) parameters[5] - 2 * (double) parameters[18]; double df22 =
    //   pow((double) parameters[18], 2) * (1 + (double) parameters[5] -
    //   (double) parameters[18] * (double) parameters[5]);

    double pacf = (1 / (double)parameters[0])
            * ((1 - (double)parameters[5]) * (acf1 / 1.6)
               + powf((double)parameters[18], 2) * (double)parameters[5] * acf2)
            / pow(1 - (double)parameters[5]
                      + (double)parameters[18] * (double)parameters[5],
                  2)
            * triplet
        + (double)parameters[4];

    REAL *current_derivatives = derivative + point_index;
    /*
        current_derivatives[0 * n_points] = (float) (-1 /
       pow((double)parameters[0], 2)) * (pf1 * acf1 + pf2 * acf2) * triplet;
        current_derivatives[1 * n_points] = (1 /  parameters[0]) * (float)(pf1 *
       (pplane1 * dDpspim + pspim1 * dDplat)); current_derivatives[2 * n_points]
       = (1 /   parameters[0]) * (float)((pf1 * ((p0t / sqrpi * pexpyt + perfyt)
       * dvxperfxt) * pspim1 / (4 * pow((double) parameters[11]*(double)
       parameters[12], 2) /  parameters[17]) + pf2 * ((p0t2 / sqrpi * pexpyt2 +
       perfyt2) * dvxperfxt2) * pspim2 / (4 * pow((double)
       parameters[11]*(double) parameters[12], 2) /  parameters[17])) *
       triplet); current_derivatives[3 * n_points] = (1 /   parameters[0]) *
       (float)((pf1 * ((p0t / sqrpi * pexpxt + perfxt) * dvyperfyt) * pspim1 /
       (4 * pow((double) parameters[11]*(double) parameters[12], 2) /
       parameters[17]) + pf2 * ((p0t2 / sqrpi * pexpxt2 + perfxt2) * dvyperfyt2)
       * pspim2 / (4 * pow((double) parameters[11]*(double) parameters[12], 2) /
       parameters[17])) * triplet); current_derivatives[4 * n_points] = 1.0;
        current_derivatives[5 * n_points] = (1 /  parameters[0]) *(float)((1 /
       dfnom) * (df21 * acf1 + df22 * acf2) * triplet); current_derivatives[6 *
       n_points] = (1 /  parameters[0]) * (float)(pf2 * (pplane2 * dDpspim2 +
       pspim2 * dDplat2) * triplet); current_derivatives[9 * n_points] =
       (float)dtripletFtrip * pacf; current_derivatives[10 * n_points] =
       (float)dtripletTtrip * pacf;
      */
    current_derivatives[0 * n_points] =
        (float)(-1 / pow((double)parameters[0], 2))
        * (pf1 * (acf1 / 1.6) + pf2 * acf2) * triplet;
    current_derivatives[1 * n_points] =
        (1 / (double)parameters[0]) * (float)(Dpspim / 1.6);
    current_derivatives[2 * n_points] = 0.0;
    current_derivatives[3 * n_points] = 0.0;
    current_derivatives[4 * n_points] = 1.0;
    current_derivatives[5 * n_points] = 0.0;
    current_derivatives[6 * n_points] = 0.0;
    current_derivatives[7 * n_points] = 0.0;
    current_derivatives[8 * n_points] = 0.0;
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