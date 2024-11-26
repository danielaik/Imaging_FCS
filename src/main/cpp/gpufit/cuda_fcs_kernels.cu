#include "cuda_fcs_kernels.cuh"

__global__ void calc_binning(const float *data, float *data1, int win_star, int hin_star, int w_temp, int h_temp,
                             int framediff, int binningX, int binningY)
{
    // Calculate global thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate

    if (idx >= w_temp || idy >= h_temp)
        return;

    for (int t = 0; t < framediff; t++)
    {
        float sum = 0.0f;
        for (int j = 0; j < binningY; j++)
        {
            int y = idy + j;
            if (y >= hin_star)
                break;
            for (int i = 0; i < binningX; i++)
            {
                int x = idx + i;
                if (x >= win_star)
                    break;

                int data_idx = t * win_star * hin_star + y * win_star + x;
                sum += data[data_idx];
            }
        }
        int data1_idx = t * w_temp * h_temp + idy * w_temp + idx;
        data1[data1_idx] = sum;
    }
}

__global__ void calc_data_bleach_correction(float *data, float *data1, int width, int height, int nopit, int ave)
{
    // function is an averaging step in temporal dimension for every ave number
    // of points, prior to performing bleach correction fitting.

    int idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();

    if ((idx < width) && (idy < height))
    {
        for (int z1 = 0; z1 < nopit; z1++)
        {
            double sum1 = 0;

            for (int yy = z1 * ave; yy < (z1 + 1) * ave; yy++)
            {
                sum1 += (float)data[yy * width * height + idy * width + idx];
            } // for yy
            data1[idy * width * nopit + idx * nopit + z1] = sum1 / ave;
        } // for z1
    } // if
}

/*
__global__ void calc_data_bleach_correction(const float *data, float *data1,
                                            int width, int height, int nopit,
                                            int ave)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate
    int z1 = blockIdx.z * blockDim.z + threadIdx.z;  // z-coordinate (time bin)

    if (idx >= width || idy >= height || z1 >= nopit)
        return;

    float sum1 = 0.0f;
    int start_time = z1 * ave;
    int end_time = start_time + ave;

    for (int yy = start_time; yy < end_time; yy++)
    {
        int data_idx = yy * width * height + idy * width + idx;
        sum1 += data[data_idx];
    }

    int data1_idx = z1 * width * height + idy * width + idx;
    data1[data1_idx] = sum1 / ave;
}
*/

__global__ void bleachcorrection(float *data, int w_temp, int h_temp, int d, int bleachcorr_order, double frametimegpu,
                                 const double *bleachcorr_params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate

    if (idx >= w_temp || idy >= h_temp)
        return;

    int pixel_idx = idy * w_temp + idx;
    int param_base = pixel_idx * bleachcorr_order;

    // Preload bleach correction parameters for this pixel into registers
    double res0 = bleachcorr_params[param_base];

    for (int i = 0; i < d; i++)
    {
        float time = (float)frametimegpu * (i + 0.5f);
        float corfunc = 0.0f;

        for (int ii = 0; ii < bleachcorr_order; ii++)
        {
            double coeff = bleachcorr_params[param_base + ii];
            corfunc += (float)(coeff * powf(time, (float)ii));
        }

        float sqrt_term = sqrtf(corfunc / res0);
        int data_idx = i * w_temp * h_temp + idy * w_temp + idx;

        data[data_idx] = data[data_idx] / sqrt_term + (float)res0 * (1.0f - sqrt_term);
    }
}

__global__ void calcacf2a(float *data, int w_temp, int h_temp, int numbin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate

    if (idx >= w_temp || idy >= h_temp)
        return;

    for (int y = 0; y < numbin; y++)
    {
        int src_idx1 = (2 * y) * w_temp * h_temp + idy * w_temp + idx;
        int src_idx2 = (2 * y + 1) * w_temp * h_temp + idy * w_temp + idx;
        int dest_idx = y * w_temp * h_temp + idy * w_temp + idx;

        data[dest_idx] = data[src_idx1] + data[src_idx2];
    }
}

__global__ void calcacf2b_NB(const float *data, int cfXDistancegpu, int cfYDistancegpu, int w, int h, int w_temp,
                             int h_temp, int pixbinX, int pixbinY, float *prod, const int *laggpu, int *prodnumarray,
                             double *NBmeanGPU, double *NBcovarianceGPU, int x, int numbin, int currentIncrement)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate

    if (idx >= w || idy >= h)
        return;

    int pixelIdx = idy * w + idx;
    int del = laggpu[x] / currentIncrement;
    int prodnum = numbin - del;

    prodnumarray[x * w * h + pixelIdx] = prodnum;

    double temp1 = 0.0;
    double temp2 = 0.0;

    // Calculate temp1 and temp2 (averages of direct and delayed monitors)
    for (int y = 0; y < prodnum; y++)
    {
        int idx1 = y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX;
        int idx2 =
            (y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp + (idx * pixbinX + cfXDistancegpu);

        float val1 = data[idx1];
        float val2 = data[idx2];

        temp1 += val1;
        temp2 += val2;

        prod[y * w * h + pixelIdx] = val1 * val2;
    }

    temp1 /= prodnum;
    temp2 /= prodnum;

    // Calculate sum of products
    double sumprod = 0.0;
    for (int y = 0; y < prodnum; y++)
    {
        sumprod += prod[y * w * h + pixelIdx];
    }

    // Store results
    NBmeanGPU[pixelIdx] = temp1;
    NBcovarianceGPU[pixelIdx] = sumprod / prodnum - temp1 * temp2;
}

__global__ void calcacf2b_ACF(const float *data, int cfXDistancegpu, int cfYDistancegpu, int w, int h, int w_temp,
                              int h_temp, int pixbinX, int pixbinY, double *data1, float *prod, const int *laggpu,
                              int *prodnumarray, const int *indexarray, double *blockvararray, double *sdarray,
                              const int *pnumgpu, int x, int numbin, int currentIncrement, int ctbin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate

    if (idx >= w || idy >= h)
        return;

    int pixelIdx = idy * w + idx;
    int del = laggpu[x] / currentIncrement;
    int prodnum = numbin - del;

    prodnumarray[x * w * h + pixelIdx] = prodnum;

    double temp1 = 0.0;
    double temp2 = 0.0;

    // First loop to calculate temp1 and temp2 (sums of direct and delayed monitors)
    for (int y = 0; y < prodnum; y++)
    {
        int idx1 = y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX;
        int idx2 =
            (y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp + (idx * pixbinX + cfXDistancegpu);

        float val1 = data[idx1];
        float val2 = data[idx2];

        temp1 += val1;
        temp2 += val2;
    }

    // Compute averages
    temp1 /= prodnum;
    temp2 /= prodnum;

    // Second loop to compute prod[...] using the averages
    for (int y = 0; y < prodnum; y++)
    {
        int idx1 = y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX;
        int idx2 =
            (y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp + (idx * pixbinX + cfXDistancegpu);

        float val1 = data[idx1];
        float val2 = data[idx2];

        prod[y * w * h + pixelIdx] = val1 * val2 - temp2 * val1 - temp1 * val2 + temp1 * temp2;
    }

    // Calculate sum of products
    double sumprod = 0.0;
    for (int y = 0; y < prodnum; y++)
    {
        sumprod += prod[y * w * h + pixelIdx];
    }

    // Store result in data1
    data1[x * w * h + pixelIdx] = sumprod / (prodnum * temp1 * temp2);

    // Blocking and variance calculations
    double sumprod2 = 0.0;
    int binct = indexarray[pixelIdx] - ctbin;
    int pnum = prodnum;

    // Blocking loop
    for (int y = 1; y <= binct; y++)
    {
        pnum = pnum / 2;
        prodnumarray[x * w * h + pixelIdx] = pnum;

        for (int z = 0; z < pnum; z++)
        {
            prod[z * w * h + pixelIdx] = (prod[2 * z * w * h + pixelIdx] + prod[(2 * z + 1) * w * h + pixelIdx]) / 2.0f;
        }
    }

    int pnumgpu_val = pnumgpu[pixelIdx];

    sumprod = 0.0;
    sumprod2 = 0.0;

    for (int z = 0; z < pnumgpu_val; z++)
    {
        double tempvariable = prod[z * w * h + pixelIdx];
        sumprod += tempvariable;
        sumprod2 += tempvariable * tempvariable;
    }

    double variance = (sumprod2 / pnumgpu_val - (sumprod / pnumgpu_val) * (sumprod / pnumgpu_val))
        / ((pnumgpu_val - 1) * temp1 * temp1 * temp2 * temp2);

    blockvararray[x * w * h + pixelIdx] = variance;
    sdarray[x * w * h + pixelIdx] = sqrt(variance);
}

/* ------------------------------------------
NOTE: SEE JCudaImageJExampleKernelcalcacf3.cu
------------------------------------------ */

__global__ void calcacf3(float *data, int cfXDistancegpu, int cfYDistancegpu, int blocklag, int w, int h, int w_temp,
                         int h_temp, int pixbinX, int pixbinY, int d, int correlatorp, int correlatorq, int chanum,
                         double frametimegpu, double *data1, float *prod, double *prodnum, double *blocksd,
                         double *upper, double *lower, int *crt, int *cr12, int *cr3, int *diffpos, double *varblock0,
                         double *varblock1, double *varblock2, double *sampgpu, int *laggpu)
{
    // this function calculates the block transformation values of the
    // intensity.

    int blocknumgpu = (int)floor(log((double)d - 1.0) / log(2.0)) - 2;
    int numbin = d; // number Of data points When they are binned
    int del; // delay Or correlation time expressed In lags
    int currentIncrement = blocklag;
    double sumprod = 0.0; // sum of all intensity products; divide by num to get
                          // the average <i(n)i(n+del)>
    double sumprod2 = 0.0; // sum of all intensity products squared; divide by
                           // num to get the average <(i(n)i(n+del))^2>
    double directm = 0.0; // direct monitor required for ACF normalization
    double delayedm = 0.0;
    int ind = 0;
    int last0 = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;
    int blockIndS = 0;

    __syncthreads();
    if ((idx < w) && (idy < h))
    {
        int x = 1;
        del = laggpu[x] / currentIncrement; // calculate the delay, i.e. the
                                            // correlation time
        for (int y = 0; y < numbin - del; y++)
        { // calculate the ...
            directm += data[y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX]; // direct And ...
            delayedm += data[(y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp
                             + (idx * pixbinX + cfXDistancegpu)]; // delayed monitor
        }
        prodnum[0] = numbin - del; // number Of correlation products
        directm /= prodnum[0]; // calculate average Of direct And delayed monitor,
        delayedm /= prodnum[0]; // i.e. the average intesity <n(0)> And <n(tau)>

        for (int y = 0; y < prodnum[0]; y++)
        { // calculate the correlation
            prod[y * w * h + idy * w + idx] =
                data[y * w_temp * h_temp + idy * pixbinY * (w + cfXDistancegpu) + idx * pixbinX]
                    * data[(y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp
                           + (idx * pixbinX + cfXDistancegpu)]
                - delayedm * data[y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX]
                - directm
                    * data[(y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp
                           + (idx * pixbinX + cfXDistancegpu)]
                + delayedm * directm;
            __syncthreads();
            sumprod += prod[y * w * h + idy * w + idx]; // calculate the sum Of prod, i.e. the raw
                                                        // correlation value ...
            sumprod2 += powf(prod[y * w * h + idy * w + idx],
                             2.0); // ... And the sum Of the squares
        }

        varblock0[idy * w + idx] = currentIncrement * frametimegpu; // the time Of the block curve
        varblock1[idy * w + idx] =
            (sumprod2 / prodnum[0] - powf(sumprod / prodnum[0], 2.0)) / (prodnum[0] * powf(directm * delayedm, 2.0));

        for (int y = 1; y < blocknumgpu; y++)
        { // perform blocking operations
            prodnum[y] = (int)floor((double)prodnum[y - 1] / 2); // the number Of samples For the blocking curve
                                                                 // decreases by a factor 2 With every Step
            sumprod = 0;
            sumprod2 = 0;
            for (int z = 0; z < prodnum[y]; z++)
            { // bin the correlation data And calculate the blocked values for
              // the SD
                prod[z * w * h + idy * w + idx] =
                    (prod[2 * z * w * h + idy * w + idx] + prod[(2 * z + 1) * w * h + idy * w + idx]) / 2;
                __syncthreads();
                sumprod += prod[z * w * h + idy * w + idx];
                sumprod2 += powf(prod[z * w * h + idy * w + idx], 2.0);
            }

            // This is the correct one
            varblock0[y * w * h + idy * w + idx] =
                (currentIncrement * powf(2, (double)y)) * frametimegpu; // the time Of the block curve
            varblock1[y * w * h + idy * w + idx] = (sumprod2 / prodnum[y] - powf(sumprod / prodnum[y], 2.0))
                / (prodnum[y] * powf(directm * delayedm, 2.0)); // value of the block curve
        }

        for (int x = 0; x < blocknumgpu; x++)
        {
            varblock1[x * w * h + idy * w + idx] =
                sqrt(varblock1[x * w * h + idy * w + idx]); // calculate the standard deviation
            varblock2[x * w * h + idy * w + idx] =
                varblock1[x * w * h + idy * w + idx] / sqrt((double)2 * (prodnum[x] - 1)); // calculate the error
            __syncthreads();
            upper[x * w * h + idy * w + idx] =
                varblock1[x * w * h + idy * w + idx] + varblock2[x * w * h + idy * w + idx]; // upper and lower quartile
            lower[x * w * h + idy * w + idx] =
                varblock1[x * w * h + idy * w + idx] - varblock2[x * w * h + idy * w + idx];
        }

        // determine index where blocking criteria are fulfilled
        for (int x = 0; x < blocknumgpu - 1; x++)
        { // do neighboring points have overlapping error bars?
            if (upper[x * w * h + idy * w + idx] > lower[(x + 1) * w * h + idy * w + idx]
                && upper[(x + 1) * w * h + idy * w + idx] > lower[x * w * h + idy * w + idx])
            {
                crt[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 2; x++)
        { // do three adjacent points have overlapping error bars?
            if (crt[x * w * h + idy * w + idx] * crt[(x + 1) * w * h + idy * w + idx] == 1)
            {
                cr12[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 1; x++)
        { // do neighboring points have a positive difference (increasing SD)?
            if (varblock1[(x + 1) * w * h + idy * w + idx] - varblock1[x * w * h + idy * w + idx] > 0)
            {
                diffpos[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 2; x++)
        { // do three neighboring points monotonically increase?
            if (diffpos[x * w * h + idy * w + idx] * diffpos[(x + 1) * w * h + idy * w + idx] == 1)
            {
                cr3[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 2; x++)
        { // find the last triple of points with monotonically increasing
          // differences and non-overlapping error bars
            if ((cr3[x * w * h + idy * w + idx] == 1 && cr12[x * w * h + idy * w + idx] == 0))
            {
                last0 = x;
            }
        }

        for (int x = 0; x <= last0; x++)
        { // indices of two pairs that pass criterion 1 an 2
            cr12[x * w * h + idy * w + idx] = 0;
        }

        cr12[(blocknumgpu - 3) * w * h + idy * w + idx] = 0; // criterion 3, the last two points can't be part of the
                                                             // blocking triple
        cr12[(blocknumgpu - 4) * w * h + idy * w + idx] = 0;

        for (int x = blocknumgpu - 5; x > 0; x--)
        { // index of triplet with overlapping error bars and after which no
          // other triplet has a significant monotonic increase
            if (cr12[x * w * h + idy * w + idx] == 1)
            { // or 4 increasing points
                ind = x + 1;
            }
        }

        if (ind == 0)
        { // if optimal blocking is not possible, use maximal blocking
            blockIndS = 0;
            if (blocknumgpu - 3 > 0)
            {
                ind = blocknumgpu - 3;
            }
            else
            {
                ind = blocknumgpu - 1;
            }
        }
        else
        {
            blockIndS = 1;
        }

        ind = (int)fmax((double)ind, (double)correlatorq - 1);
        data1[idy * w + idx] = (double)ind;
        data1[w * h + idy * w + idx] = (double)blockIndS;

    } // if ((idx < w) && (idy < h))
} // calcacf3