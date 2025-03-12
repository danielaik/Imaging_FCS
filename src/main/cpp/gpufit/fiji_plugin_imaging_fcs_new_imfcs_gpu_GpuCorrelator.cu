#include <algorithm>
#include <vector>

#include "acf_input_params_wrapper.h"
#include "cuda_fcs_kernels.cuh"
#include "cuda_utils/cuda_device_ptr.h"
#include "definitions.h"
#include "fiji_plugin_imaging_fcs_new_imfcs_gpu_GpuCorrelator.h"
#include "java_array.h"

void NBCalculationFunction(JNIEnv *env, ACFInputParamsWrapper &p, jfloat *Cpixels, jdouble *Cbleachcorr_p,
                           jdouble *Csamp, jint *Clag, jdoubleArray NBmeanGPU, jdoubleArray NBcovarianceGPU)
{
    // Initialize parameters
    int framediff = p.lastframe - p.firstframe + 1;
    size_t size_pixels = static_cast<size_t>(p.w_temp) * p.h_temp * framediff;
    size_t size_bleachcorr_p = static_cast<size_t>(p.w_temp) * p.h_temp * p.bleachcorr_order;
    size_t size2 = static_cast<size_t>(framediff) * p.width * p.height;
    size_t size_prodnumarray = static_cast<size_t>(p.chanum) * p.width * p.height;

    // Host arrays
    std::vector<double> CNBmeanGPU(p.width * p.height, 0.0);
    std::vector<double> CNBcovarianceGPU(p.width * p.height, 0.0);
    std::vector<float> prod(size2, 0.0f); // Initialize prod
    std::vector<int> prodnumarray(size_prodnumarray, 0); // Initialize prodnumarray

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

    // Device arrays
    cuda_utils::CudaDevicePtr<float> d_Cpixels(size_pixels);
    cuda_utils::CudaDevicePtr<double> d_Cbleachcorr_p(size_bleachcorr_p);
    cuda_utils::CudaDevicePtr<int> d_Clag(p.chanum);
    cuda_utils::CudaDevicePtr<double> d_NBmeanGPU(p.width * p.height);
    cuda_utils::CudaDevicePtr<double> d_NBcovarianceGPU(p.width * p.height);
    cuda_utils::CudaDevicePtr<float> d_prod(size2);
    cuda_utils::CudaDevicePtr<int> d_prodnumarray(size_prodnumarray);

    // Copy data to device
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_Cpixels.get(), Cpixels, size_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Cbleachcorr_p.get(), Cbleachcorr_p, size_bleachcorr_p * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Clag.get(), Clag, p.chanum * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_prod.get(), prod.data(), size2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_prodnumarray.get(), prodnumarray.data(), size_prodnumarray * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    // Copy N & B arrays to device (initialized to zero)
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_NBmeanGPU.get(), CNBmeanGPU.data(), CNBmeanGPU.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_NBcovarianceGPU.get(), CNBcovarianceGPU.data(),
                                      CNBcovarianceGPU.size() * sizeof(double), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Configure CUDA grid and block sizes
    int BLKSIZEXY = 16;
    int max_dim = std::max(p.width, p.height);
    int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;
    dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
    dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

    int max_dim_input = std::max(p.w_temp, p.h_temp);
    int GRIDSIZEXY_Input = (max_dim_input + BLKSIZEXY - 1) / BLKSIZEXY;
    dim3 gridSize_Input(GRIDSIZEXY_Input, GRIDSIZEXY_Input, 1);

    // Run bleach correction kernel if needed
    if (p.bleachcorr_gpu)
    {
        bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(
            d_Cpixels.get(), p.w_temp, p.h_temp, framediff, p.bleachcorr_order, p.frametime, d_Cbleachcorr_p.get());
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());
    }

    int numbin = framediff;
    int currentIncrement = 1;
    int ctbin = 0;

    // Ensure Csamp has enough elements
    if (p.chanum < 2)
    {
        throw std::runtime_error("Csamp array does not have enough elements for N&B calculation.");
    }

    for (int x = 1; x < 2; x++)
    {
        if (currentIncrement != static_cast<int>(Csamp[x]))
        {
            numbin = static_cast<int>(std::floor(static_cast<double>(numbin) / 2.0));
            currentIncrement = static_cast<int>(Csamp[x]);
            ctbin++;
            calcacf2a<<<gridSize_Input, blockSize, 0, stream>>>(d_Cpixels.get(), p.w_temp, p.h_temp, numbin);
            CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
            CUDA_CHECK_STATUS(cudaGetLastError());
        }

        // Launch calcacf2b_NB kernel with proper arguments
        calcacf2b_NB<<<gridSize, blockSize, 0, stream>>>(
            d_Cpixels.get(), p.cfXDistance, p.cfYDistance, p.width, p.height, p.w_temp, p.h_temp, p.pixbinX, p.pixbinY,
            d_prod.get(), d_Clag.get(), d_prodnumarray.get(), d_NBmeanGPU.get(), d_NBcovarianceGPU.get(), x, numbin,
            currentIncrement);

        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());
    }

    // Copy results back to host
    CUDA_CHECK_STATUS(cudaMemcpyAsync(CNBmeanGPU.data(), d_NBmeanGPU.get(), CNBmeanGPU.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(CNBcovarianceGPU.data(), d_NBcovarianceGPU.get(),
                                      CNBcovarianceGPU.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Destroy CUDA stream
    CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

    // Copy results back to Java arrays
    env->SetDoubleArrayRegion(NBmeanGPU, 0, CNBmeanGPU.size(), CNBmeanGPU.data());
    env->SetDoubleArrayRegion(NBcovarianceGPU, 0, CNBcovarianceGPU.size(), CNBcovarianceGPU.data());
}

void ACFCalculationFunction(JNIEnv *env, ACFInputParamsWrapper &p, jfloat *Cpixels, jdouble *Cbleachcorr_p,
                            jdouble *Csamp, jint *Clag, jdoubleArray pixels1, jdoubleArray blockvararray,
                            jdoubleArray blocked1D)
{
    // Initialize parameters
    int framediff = p.lastframe - p.firstframe + 1;
    size_t size_pixels = static_cast<size_t>(p.w_temp) * p.h_temp * framediff;
    size_t size1 = static_cast<size_t>(p.width) * p.height * p.chanum;
    size_t size2 = static_cast<size_t>(framediff) * p.width * p.height;
    size_t sizeblockvararray = static_cast<size_t>(p.chanum) * p.width * p.height;
    size_t size_bleachcorr_p = static_cast<size_t>(p.w_temp) * p.h_temp * p.bleachcorr_order;

    int blocknumgpu = static_cast<int>(std::floor(std::log(p.mtab1) / std::log(2.0)) - 2);

    // Configure CUDA grid and block sizes
    int BLKSIZEXY = 16;
    int max_dim = std::max(p.width, p.height);
    int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;
    dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
    dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

    int max_dim_input = std::max(p.w_temp, p.h_temp);
    int GRIDSIZEXY_Input = (max_dim_input + BLKSIZEXY - 1) / BLKSIZEXY;
    dim3 gridSize_Input(GRIDSIZEXY_Input, GRIDSIZEXY_Input, 1);

    // Host arrays
    std::vector<double> Cpixels1(size1);
    std::vector<float> prod(size2);
    std::vector<double> Cblocked1D(size1);
    std::vector<double> prodnum(blocknumgpu, 1.0);
    std::vector<double> upper(blocknumgpu * p.width * p.height, 0.0);
    std::vector<double> lower(blocknumgpu * p.width * p.height, 0.0);
    std::vector<double> varblock0(blocknumgpu * p.width * p.height, 0.0);
    std::vector<double> varblock1(blocknumgpu * p.width * p.height, 0.0);
    std::vector<double> varblock2(blocknumgpu * p.width * p.height, 0.0);

    std::vector<int> prodnumarray(p.chanum * p.width * p.height);
    std::vector<int> indexarray(p.width * p.height);
    std::vector<double> Cblockvararray(sizeblockvararray);
    std::vector<double> blocksdarray(sizeblockvararray);
    std::vector<int> pnumgpu(p.width * p.height);

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

    // Device arrays
    cuda_utils::CudaDevicePtr<float> d_Cpixels(size_pixels);
    cuda_utils::CudaDevicePtr<double> d_Cpixels1(size1);
    cuda_utils::CudaDevicePtr<double> d_Cbleachcorr_p(size_bleachcorr_p);
    cuda_utils::CudaDevicePtr<int> d_Clag(p.chanum);
    cuda_utils::CudaDevicePtr<float> d_prod(size2);
    cuda_utils::CudaDevicePtr<double> d_prodnum(prodnum.size());
    cuda_utils::CudaDevicePtr<double> d_upper(upper.size());
    cuda_utils::CudaDevicePtr<double> d_lower(lower.size());
    cuda_utils::CudaDevicePtr<double> d_varblock0(varblock0.size());
    cuda_utils::CudaDevicePtr<double> d_varblock1(varblock1.size());
    cuda_utils::CudaDevicePtr<double> d_varblock2(varblock2.size());

    // Copy data to device
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_Cpixels.get(), Cpixels, size_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Cbleachcorr_p.get(), Cbleachcorr_p, size_bleachcorr_p * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Clag.get(), Clag, p.chanum * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_prodnum.get(), prodnum.data(), prodnum.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_upper.get(), upper.data(), upper.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_lower.get(), lower.data(), lower.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_varblock0.get(), varblock0.data(), varblock0.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_varblock1.get(), varblock1.data(), varblock1.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_varblock2.get(), varblock2.data(), varblock2.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Run kernels
    if (p.bleachcorr_gpu)
    {
        bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(
            d_Cpixels.get(), p.w_temp, p.h_temp, framediff, p.bleachcorr_order, p.frametime, d_Cbleachcorr_p.get());
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());
    }

    calcacf3<<<gridSize, blockSize, 0, stream>>>(
        d_Cpixels.get(), p.cfXDistance, p.cfYDistance, 1, p.width, p.height, p.w_temp, p.h_temp, p.pixbinX, p.pixbinY,
        framediff, static_cast<int>(p.correlatorq), p.frametime, d_Cpixels1.get(), d_prod.get(), d_prodnum.get(),
        d_upper.get(), d_lower.get(), d_varblock0.get(), d_varblock1.get(), d_varblock2.get(), d_Clag.get());

    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
    CUDA_CHECK_STATUS(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(Cpixels1.data(), d_Cpixels1.get(), size1 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Copy Cpixels1 to Cblocked1D
    std::copy(Cpixels1.begin(), Cpixels1.end(), Cblocked1D.begin());

    // Initialize indexarray and pnumgpu
    size_t counter_indexarray = 0;
    for (int y = 0; y < p.height; y++)
    {
        for (int x = 0; x < p.width; x++)
        {
            int idx = y * p.width + x;
            indexarray[counter_indexarray] = static_cast<int>(Cpixels1[idx]);

            // Minimum number of products
            double tempval = indexarray[counter_indexarray] - std::log2(p.sampchanumminus1);
            if (tempval < 0)
            {
                tempval = 0;
            }
            pnumgpu[counter_indexarray] = static_cast<int>(std::floor(p.mtabchanumminus1 / std::pow(2.0, tempval)));
            counter_indexarray++;
        }
    }

    // Re-copy Cpixels to device if modified
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_Cpixels.get(), Cpixels, size_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Device arrays for calcacf2
    cuda_utils::CudaDevicePtr<int> d_prodnumarray(p.chanum * p.width * p.height);
    cuda_utils::CudaDevicePtr<int> d_indexarray(p.width * p.height);
    cuda_utils::CudaDevicePtr<double> d_Cblockvararray(sizeblockvararray);
    cuda_utils::CudaDevicePtr<double> d_blocksdarray(sizeblockvararray);
    cuda_utils::CudaDevicePtr<int> d_pnumgpu(p.width * p.height);

    // Copy data for calcacf2
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_prodnumarray.get(), prodnumarray.data(), prodnumarray.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_indexarray.get(), indexarray.data(), indexarray.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Cblockvararray.get(), Cblockvararray.data(),
                                      Cblockvararray.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(d_blocksdarray.get(), blocksdarray.data(), blocksdarray.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(d_pnumgpu.get(), pnumgpu.data(), pnumgpu.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Run calcacf2 kernels
    if (p.bleachcorr_gpu)
    {
        bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(
            d_Cpixels.get(), p.w_temp, p.h_temp, framediff, p.bleachcorr_order, p.frametime, d_Cbleachcorr_p.get());
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());
    }

    int numbin = framediff;
    int currentIncrement = 1;
    int ctbin = 0;

    for (int x = 0; x < p.chanum; x++)
    {
        if (currentIncrement != Csamp[x])
        {
            numbin = static_cast<int>(std::floor(static_cast<double>(numbin) / 2.0));
            currentIncrement = static_cast<int>(Csamp[x]);
            ctbin++;
            calcacf2a<<<gridSize_Input, blockSize, 0, stream>>>(d_Cpixels.get(), p.w_temp, p.h_temp, numbin);
            CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
            CUDA_CHECK_STATUS(cudaGetLastError());
        }

        // Launch calcacf2b_ACF
        calcacf2b_ACF<<<gridSize, blockSize, 0, stream>>>(
            d_Cpixels.get(), p.cfXDistance, p.cfYDistance, p.width, p.height, p.w_temp, p.h_temp, p.pixbinX, p.pixbinY,
            d_Cpixels1.get(), d_prod.get(), d_Clag.get(), d_prodnumarray.get(), d_indexarray.get(),
            d_Cblockvararray.get(), d_blocksdarray.get(), d_pnumgpu.get(), x, numbin, currentIncrement, ctbin);

        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
    }

    // Copy results back to host
    CUDA_CHECK_STATUS(
        cudaMemcpyAsync(Cpixels1.data(), d_Cpixels1.get(), size1 * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_STATUS(cudaMemcpyAsync(Cblockvararray.data(), d_Cblockvararray.get(),
                                      Cblockvararray.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

    // Destroy CUDA stream
    CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

    // Copy results back to Java arrays
    env->SetDoubleArrayRegion(pixels1, 0, size1, Cpixels1.data());
    env->SetDoubleArrayRegion(blockvararray, 0, Cblockvararray.size(), Cblockvararray.data());
    env->SetDoubleArrayRegion(blocked1D, 0, size1, Cblocked1D.data());
}

void JNICALL Java_fiji_plugin_imaging_1fcs_new_1imfcs_gpu_GpuCorrelator_calcACF(
    JNIEnv *env, jclass cls, jfloatArray pixels, jdoubleArray pixels1, jdoubleArray blockvararray,
    jdoubleArray NBmeanGPU, jdoubleArray NBcovarianceGPU, jdoubleArray blocked1D, jdoubleArray bleachcorr_p,
    jdoubleArray samp, jintArray lag, jobject ACFInputParams)
{
    try
    {
        // Wrap Java arrays with JavaArray class
        JavaArray<jfloatArray, jfloat> Cpixels(env, pixels);
        JavaArray<jdoubleArray, jdouble> Cbleachcorr_p(env, bleachcorr_p);
        JavaArray<jdoubleArray, jdouble> Csamp(env, samp);
        JavaArray<jintArray, jint> Clag(env, lag);

        // Retrieve parameters using the wrapper class
        ACFInputParamsWrapper p(env, ACFInputParams);

        if (p.isNBcalculation)
        {
            NBCalculationFunction(env, p, Cpixels.elements(), Cbleachcorr_p.elements(), Csamp.elements(),
                                  Clag.elements(), NBmeanGPU, NBcovarianceGPU);
        }
        else
        {
            ACFCalculationFunction(env, p, Cpixels.elements(), Cbleachcorr_p.elements(), Csamp.elements(),
                                   Clag.elements(), pixels1, blockvararray, blocked1D);
        }
    }
    catch (const std::exception &e)
    {
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}
