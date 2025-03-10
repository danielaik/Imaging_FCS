#include <vector>

#include "acf_input_params_wrapper.h"
#include "cuda_fcs_kernels.cuh"
#include "cuda_utils/cuda_device_ptr.h"
#include "fiji_plugin_imaging_fcs_new_imfcs_gpu_GpuParameters.h"
#include "gpufit.h"
#include "java_array.h"

jboolean JNICALL Java_fiji_plugin_imaging_1fcs_new_1imfcs_gpu_GpuParameters_isBinningMemorySufficient(
    JNIEnv *env, jclass cls, jobject ACFInputParams)
{
    try
    {
        // Retrieve parameters using the wrapper class
        ACFInputParamsWrapper p(env, ACFInputParams);

        // Calculate the maximum memory required for binning
        int framediff = p.lastframe - p.firstframe + 1;
        size_t required_memory =
            static_cast<size_t>(p.win_star) * p.hin_star + static_cast<size_t>(p.w_temp) * p.h_temp;
        required_memory *= static_cast<size_t>(framediff) * sizeof(float);

        // Get the free and total device memory
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK_STATUS(cudaMemGetInfo(&free_bytes, &total_bytes));

        // Check if required memory exceeds 90% of free memory
        if (required_memory > static_cast<size_t>(free_bytes * 0.9))
        {
            return JNI_FALSE;
        }
        else
        {
            return JNI_TRUE;
        }
    }
    catch (std::runtime_error &e)
    {
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
        return JNI_FALSE;
    }
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    calcBinning
 * Signature: ([F[FLgpufitImFCS/GpufitImFCS/ACFParameters;)V
 */
void JNICALL Java_fiji_plugin_imaging_1fcs_new_1imfcs_gpu_GpuParameters_calcBinning(JNIEnv *env, jclass cls,
                                                                                    jfloatArray indata,
                                                                                    jfloatArray outdata,
                                                                                    jobject ACFInputParams)
{
    try
    {
        JavaArray<jfloatArray, jfloat> Cindata(env, indata);

        // Retrieve parameters using the wrapper class
        ACFInputParamsWrapper p(env, ACFInputParams);

        // Compute frame difference
        int framediff = p.lastframe - p.firstframe + 1;

        // Configure CUDA grid and block sizes
        int BLKSIZEXY = 16;
        int max_dim = std::max(p.w_temp, p.h_temp);
        int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;

        dim3 blockSizeBin(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSizeBin(GRIDSIZEXY, GRIDSIZEXY, 1);

        // Calculate sizes for device and host memory
        size_t num_input_elements = static_cast<size_t>(p.win_star) * p.hin_star * framediff;
        size_t num_output_elements = static_cast<size_t>(p.w_temp) * p.h_temp * framediff;

        // Allocate device memory using cuda_utils::CudaDevicePtr
        cuda_utils::CudaDevicePtr<float> d_Cindata(num_input_elements);
        cuda_utils::CudaDevicePtr<float> d_Coutput(num_output_elements);

        // Allocate host memory using std::vector
        std::vector<float> Coutput(num_output_elements);

        // Copy to GPU
        CUDA_CHECK_STATUS(cudaMemcpy(d_Cindata.get(), Cindata.elements(), num_input_elements * sizeof(float),
                                     cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        calc_binning<<<gridSizeBin, blockSizeBin, 0, stream>>>(d_Cindata.get(), d_Coutput.get(), p.win_star, p.hin_star,
                                                               p.w_temp, p.h_temp, framediff, p.binningX, p.binningY);

        // Synchronize and check for errors
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());

        // copy memory from device to host
        CUDA_CHECK_STATUS(
            cudaMemcpy(Coutput.data(), d_Coutput.get(), num_output_elements * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        // copy values to Java output arrays.
        env->SetFloatArrayRegion(outdata, 0, num_output_elements, Coutput.data());
    }
    catch (std::runtime_error &e)
    {
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}

void JNICALL Java_fiji_plugin_imaging_1fcs_new_1imfcs_gpu_GpuParameters_calcDataBleachCorrection(
    JNIEnv *env, jclass cls, jfloatArray pixels, jfloatArray outdata, jobject ACFInputParams)
{
    try
    {
        JavaArray<jfloatArray, jfloat> Cpixels(env, pixels);

        // Retrieve parameters using the wrapper class
        ACFInputParamsWrapper p(env, ACFInputParams);

        // Compute sizes
        int framediff = p.lastframe - p.firstframe + 1;
        size_t num_elements_input = static_cast<size_t>(p.w_temp) * p.h_temp * framediff;
        size_t num_elements_output = static_cast<size_t>(p.w_temp) * p.h_temp * p.nopit;

        // Configure CUDA grid and block sizes
        int BLKSIZEXY = 16;
        int max_dim = std::max(p.w_temp, p.h_temp);
        int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;
        dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

        // Allocate memory on GPU
        cuda_utils::CudaDevicePtr<float> d_Cpixels(num_elements_input);
        cuda_utils::CudaDevicePtr<float> d_Coutput(num_elements_output);

        // Allocate host memory using std::vector
        std::vector<float> Coutput(num_elements_output);

        // Copy to GPU
        CUDA_CHECK_STATUS(cudaMemcpy(d_Cpixels.get(), Cpixels.elements(), num_elements_input * sizeof(float),
                                     cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        // Launch CUDA kernel
        calc_data_bleach_correction<<<gridSize, blockSize, 0, stream>>>(d_Cpixels.get(), d_Coutput.get(), p.w_temp,
                                                                        p.h_temp, p.nopit, p.ave);

        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());

        // copy memory from device to host
        CUDA_CHECK_STATUS(
            cudaMemcpy(Coutput.data(), d_Coutput.get(), num_elements_output * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        // copy values to Java output arrays.
        env->SetFloatArrayRegion(outdata, 0, num_elements_output, Coutput.data());
    }
    catch (std::runtime_error &e)
    {
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}