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

__global__ void calcacf3(float *data, int cfXDistancegpu, int cfYDistancegpu, int blocklag, int w, int h, int w_temp,
                         int h_temp, int pixbinX, int pixbinY, int d, int correlatorq, double frametimegpu,
                         double *data1, float *prod, double *prodnum, double *upper, double *lower, double *varblock0,
                         double *varblock1, double *varblock2, int *laggpu)
{
    // this function calculates the block transformation values of the
    // intensity.

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate

    if (idx >= w || idy >= h)
        return;

    int pixelIdx = idy * w + idx;

    int blocknumgpu = (int)floor(log2((double)d - 1.0)) - 2;
    int numbin = d;
    int del = laggpu[1] / blocklag; // Since x is fixed at 1

    // Initialize variables for direct and delayed monitors
    double directm = 0.0;
    double delayedm = 0.0;

    int prodnum0 = numbin - del;
    prodnum[0] = prodnum0;

    // Calculate directm and delayedm
    for (int y = 0; y < prodnum0; y++)
    {
        int idx_direct = y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX;
        int idx_delayed =
            (y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp + (idx * pixbinX + cfXDistancegpu);

        float val_direct = data[idx_direct];
        float val_delayed = data[idx_delayed];

        directm += val_direct;
        delayedm += val_delayed;
    }

    directm /= prodnum0;
    delayedm /= prodnum0;

    double sumprod = 0.0;
    double sumprod2 = 0.0;

    // Calculate prod, sumprod, and sumprod2
    for (int y = 0; y < prodnum0; y++)
    {
        int idx_direct = y * w_temp * h_temp + idy * pixbinY * w_temp + idx * pixbinX;
        int idx_delayed =
            (y + del) * w_temp * h_temp + (idy * pixbinY + cfYDistancegpu) * w_temp + (idx * pixbinX + cfXDistancegpu);

        float val_direct = data[idx_direct];
        float val_delayed = data[idx_delayed];

        float prod_val = val_direct * val_delayed - delayedm * val_direct - directm * val_delayed + directm * delayedm;

        prod[y * w * h + pixelIdx] = prod_val;

        sumprod += prod_val;
        sumprod2 += prod_val * prod_val;
    }

    // Compute variance for initial block
    double variance = (sumprod2 / prodnum0 - (sumprod / prodnum0) * (sumprod / prodnum0))
        / (prodnum0 * directm * directm * delayedm * delayedm);

    varblock0[pixelIdx] = blocklag * frametimegpu; // Time of the block curve
    varblock1[pixelIdx] = variance;

    // Blocking operations
    for (int y = 1; y < blocknumgpu; y++)
    {
        prodnum[y] = prodnum[y - 1] / 2;
        int pnum = prodnum[y];
        sumprod = 0.0;
        sumprod2 = 0.0;

        for (int z = 0; z < pnum; z++)
        {
            int idx1 = 2 * z * w * h + pixelIdx;
            int idx2 = (2 * z + 1) * w * h + pixelIdx;

            prod[z * w * h + pixelIdx] = (prod[idx1] + prod[idx2]) / 2.0f;

            double prod_val = prod[z * w * h + pixelIdx];
            sumprod += prod_val;
            sumprod2 += prod_val * prod_val;
        }

        variance =
            (sumprod2 / pnum - (sumprod / pnum) * (sumprod / pnum)) / (pnum * directm * directm * delayedm * delayedm);

        varblock0[y * w * h + pixelIdx] = blocklag * pow(2.0, (double)y) * frametimegpu;
        varblock1[y * w * h + pixelIdx] = variance;
    }

    // Compute standard deviation and error estimates
    for (int x = 0; x < blocknumgpu; x++)
    {
        double stddev = sqrt(varblock1[x * w * h + pixelIdx]);
        varblock1[x * w * h + pixelIdx] = stddev;

        double error = stddev / sqrt(2.0 * (prodnum[x] - 1));
        varblock2[x * w * h + pixelIdx] = error;

        upper[x * w * h + pixelIdx] = stddev + error;
        lower[x * w * h + pixelIdx] = stddev - error;
    }

    // Assume that idx and idy are already defined and within bounds
    int last_index_meeting_criteria = -1;
    int index = 0;
    int blockIndS = 0; // Indicator for whether optimal blocking is possible

    // Loop over blocks to determine the last index meeting the criteria
    for (int x = 0; x <= blocknumgpu - 3; x++)
    {
        // Check if neighboring points have overlapping error bars
        bool overlap1 = upper[x * w * h + pixelIdx] > lower[(x + 1) * w * h + pixelIdx]
            && upper[(x + 1) * w * h + pixelIdx] > lower[x * w * h + pixelIdx];
        bool overlap2 = upper[(x + 1) * w * h + pixelIdx] > lower[(x + 2) * w * h + pixelIdx]
            && upper[(x + 2) * w * h + pixelIdx] > lower[(x + 1) * w * h + pixelIdx];
        bool overlap = overlap1 && overlap2;

        // Check if variance values are increasing
        bool increasing1 = varblock1[(x + 1) * w * h + pixelIdx] - varblock1[x * w * h + pixelIdx] > 0;
        bool increasing2 = varblock1[(x + 2) * w * h + pixelIdx] - varblock1[(x + 1) * w * h + pixelIdx] > 0;
        bool isIncreasing = increasing1 && increasing2;

        // Update last_index_meeting_criteria if conditions are met
        if (!overlap && isIncreasing)
        {
            last_index_meeting_criteria = x;
        }
    }

    // If a suitable index was found, search for overlapping error bars
    if (last_index_meeting_criteria != -1)
    {
        for (int x = last_index_meeting_criteria + 1; x <= blocknumgpu - 3; x++)
        {
            bool overlap1 = upper[x * w * h + pixelIdx] > lower[(x + 1) * w * h + pixelIdx]
                && upper[(x + 1) * w * h + pixelIdx] > lower[x * w * h + pixelIdx];
            bool overlap2 = upper[(x + 1) * w * h + pixelIdx] > lower[(x + 2) * w * h + pixelIdx]
                && upper[(x + 2) * w * h + pixelIdx] > lower[(x + 1) * w * h + pixelIdx];
            bool overlap = overlap1 && overlap2;

            if (overlap)
            {
                index = x + 1;
                break;
            }
        }
    }

    // Determine if optimal blocking is possible
    if (index == 0)
    {
        // Optimal blocking is not possible; use maximal blocking
        blockIndS = 0;
        if (blocknumgpu - 3 > 0)
        {
            index = blocknumgpu - 3; // Use the third-last point if it exists
        }
        else
        {
            index = blocknumgpu - 1;
        }
    }
    else
    {
        blockIndS = 1;
    }

    // Ensure the index is at least correlatorq - 1
    index = max(index, correlatorq - 1);

    // Store the results
    data1[idy * w + idx] = (double)index;
    data1[w * h + idy * w + idx] = (double)blockIndS;
} // calcacf3