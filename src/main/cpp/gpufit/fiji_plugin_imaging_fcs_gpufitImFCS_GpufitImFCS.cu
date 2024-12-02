#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "cuda_fcs_kernels.cuh"
#include "cuda_utils/cuda_device_ptr.h"
#include "definitions.h"
#include "fiji_plugin_imaging_fcs_gpufitImFCS_GpufitImFCS.h"
#include "gpufit.h"

/* -------------------------------------------------------------------------------------------------------
* from com_github_gpufit_Gpufit.cpp START
* NOTE: creates the gpufitJNI.dll. File is located at
/Gpufit/Gpufit/java/adapter/
-------------------------------------------------------------------------------------------------------
*/
void *buffer_address(JNIEnv *env, jobject buffer)
{
    if (buffer == 0)
    {
        return 0;
    }
    else
    {
        return env->GetDirectBufferAddress(buffer);
    }
}

/*
 * Calls gpufit(), no consistency checks on this side.
 *
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    fit
 * Signature:
 * (IILjava/nio/FloatBuffer;Ljava/nio/FloatBuffer;ILjava/nio/FloatBuffer;FILjava/nio/IntBuffer;IILjava/nio/ByteBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)I
 */
jint JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_fit(
    JNIEnv *env, jclass cls, jint number_fits, jint number_points, jobject data_buffer, jobject weights_buffer,
    jint model_id, jobject initial_parameter_buffer, jfloat tolerance, jint max_number_iterations, jint num_valid_coefs,
    jobject paramters_to_fit_buffer, jint estimator_id, jint user_info_size, jobject user_info_buffer,
    jobject output_parameters_buffer, jobject output_states_buffer, jobject output_chi_squares_buffer,
    jobject output_number_iterations_buffer)
{
    // get pointer to buffers
    REAL *data = (REAL *)buffer_address(env, data_buffer);
    REAL *weights = (REAL *)buffer_address(env, weights_buffer);
    REAL *initial_parameters = (REAL *)buffer_address(env, initial_parameter_buffer);
    int *parameters_to_fit = (int *)buffer_address(env, paramters_to_fit_buffer);
    char *user_info = (char *)buffer_address(env, user_info_buffer);
    REAL *output_parameters = (REAL *)buffer_address(env, output_parameters_buffer);
    int *output_states = (int *)buffer_address(env, output_states_buffer);
    REAL *output_chi_squares = (REAL *)buffer_address(env, output_chi_squares_buffer);
    int *output_number_iterations = (int *)buffer_address(env, output_number_iterations_buffer);

    // call to gpufit
    // NOTE: Added num_valid_coefs
    int status = gpufit(number_fits, number_points, data, weights, model_id, initial_parameters, tolerance,
                        max_number_iterations, num_valid_coefs, parameters_to_fit, estimator_id, user_info_size,
                        user_info, output_parameters, output_states, output_chi_squares, output_number_iterations);

    // lmfit_cuda

    return status;
}

/*
 * Calls gpufit_get_last_error()
 *
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    getLastError
 * Signature: ()Ljava/lang/String;
 */
jstring JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_getLastError(JNIEnv *env, jclass cls)
{
    char const *error = gpufit_get_last_error();
    return env->NewStringUTF(error);
}

/*
 * Calls gpufit_cuda_available()
 *
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    isCudaAvailableInt
 * Signature: ()Z
 */
jboolean JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_isCudaAvailableInt(JNIEnv *env, jclass cls)
{
    return gpufit_cuda_available() == 1 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_fiji_plugin_imaging_1fcs_new_1imfcs_model_HardwareModel_isCudaAvailableInt(JNIEnv *env,
                                                                                                           jclass cls)
{
    return gpufit_cuda_available() == 1 ? JNI_TRUE : JNI_FALSE;
}

/*
 * Calls gpufit_get_cuda_version()
 *
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    getCudaVersionAsArray
 * Signature: ()[I
 */
jintArray JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_getCudaVersionAsArray(JNIEnv *env, jclass cls)
{
    int runtime_version, driver_version;
    if (gpufit_get_cuda_version(&runtime_version, &driver_version) == ReturnState::OK)
    {
        // create int[2] in Java and fill with values
        jintArray array = env->NewIntArray(2);
        jint fill[2];
        fill[0] = runtime_version;
        fill[1] = driver_version;
        env->SetIntArrayRegion(array, 0, 2, fill);
        return array;
    }
    else
    {
        return 0;
    }
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    resetGPU
 * Signature: ()V
 */
void JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_resetGPU(JNIEnv *env, jclass cls)
{
    try
    {
        cudaDeviceReset();
    }
    catch (std::runtime_error &e)
    {
        // see: https://www.rgagnon.com/javadetails/java-0323.html
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    calcDataBleachCorrection
 * Signature: ([F[FLgpufitImFCS/GpufitImFCS/ACFParameters;)V
 */
void JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_calcDataBleachCorrection(JNIEnv *env, jclass cls,
                                                                                            jfloatArray pixels,
                                                                                            jfloatArray outdata,
                                                                                            jobject ACFInputParams)
{
    // input arrays required for calculations.
    jfloat *Cpixels = nullptr;

    try
    {
        Cpixels = env->GetFloatArrayElements(pixels, NULL);
        if (Cpixels == nullptr)
        {
            throw std::runtime_error("Failed to access pixel data from Java array");
        }

        // get parameters from the ACFInputParams object
        // we need width, height, cfXDistancegpu, cfYDistancegpu, nopit, ave
        // we also need firstframe and lastframe for setting blockSize and
        // gridSize
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID firstframeId = env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId = env->GetFieldID(ACFInputParamsCls, "lastframe", "I");
        jfieldID nopitId = env->GetFieldID(ACFInputParamsCls, "nopit", "I");
        jfieldID aveId = env->GetFieldID(ACFInputParamsCls, "ave", "I");

        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);
        jint nopit = env->GetIntField(ACFInputParams, nopitId);
        jint ave = env->GetIntField(ACFInputParams, aveId);

        // Compute sizes
        int framediff = lastframe - firstframe + 1;
        size_t num_elements_input = static_cast<size_t>(w_temp) * h_temp * framediff;
        size_t num_elements_output = static_cast<size_t>(w_temp) * h_temp * nopit;

        // Configure CUDA grid and block sizes
        int BLKSIZEXY = 16;
        int max_dim = std::max(w_temp, h_temp);
        int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;
        dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

        // Allocate memory on GPU
        cuda_utils::CudaDevicePtr<float> d_Cpixels(num_elements_input);
        cuda_utils::CudaDevicePtr<float> d_Coutput(num_elements_output);

        // Allocate host memory using std::vector
        std::vector<float> Coutput(num_elements_output);

        // Copy to GPU
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cpixels.get(), Cpixels, num_elements_input * sizeof(float), cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        // Launch CUDA kernel
        calc_data_bleach_correction<<<gridSize, blockSize, 0, stream>>>(d_Cpixels.get(), d_Coutput.get(), w_temp,
                                                                        h_temp, nopit, ave);

        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());

        // copy memory from device to host
        CUDA_CHECK_STATUS(
            cudaMemcpy(Coutput.data(), d_Coutput.get(), num_elements_output * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        CUDA_CHECK_STATUS(cudaDeviceReset());

        // copy values to Java output arrays.
        env->SetFloatArrayRegion(outdata, 0, num_elements_output, Coutput.data());

        // release resources
        env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
    }
    catch (std::runtime_error &e)
    {
        // Release Java array elements if they were acquired
        if (Cpixels != nullptr)
        {
            env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
        }

        // Throw the exception to Java
        // see: https://www.rgagnon.com/javadetails/java-0323.html
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    isBinningMemorySufficient
 * Signature: (LgpufitImFCS/GpufitImFCS/ACFParameters;)Z
 */
jboolean JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_isBinningMemorySufficient(JNIEnv *env,
                                                                                                 jclass cls,
                                                                                                 jobject ACFInputParams)
{
    try
    {
        // get parameters from the ACFInputParams object
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);
        jfieldID win_starId = env->GetFieldID(ACFInputParamsCls, "win_star", "I");
        jfieldID hin_starId = env->GetFieldID(ACFInputParamsCls, "hin_star", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID firstframeId = env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId = env->GetFieldID(ACFInputParamsCls, "lastframe", "I");

        jint win_star = env->GetIntField(ACFInputParams, win_starId);
        jint hin_star = env->GetIntField(ACFInputParams, hin_starId);
        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);

        // Calculate the maximum memory required for binning
        int framediff = lastframe - firstframe + 1;
        size_t required_memory = static_cast<size_t>(win_star) * hin_star + static_cast<size_t>(w_temp) * h_temp;
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
void JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_calcBinning(JNIEnv *env, jclass cls,
                                                                               jfloatArray indata, jfloatArray outdata,
                                                                               jobject ACFInputParams)
{
    jfloat *Cindata = nullptr;

    try
    {
        Cindata = env->GetFloatArrayElements(indata, NULL);
        if (Cindata == nullptr)
        {
            throw std::runtime_error("Failed to access input data from Java array");
        }

        // get parameters from the ACFInputParams object
        // we need w_temp, h_temp, binningX, binningY
        // we also need firstframe and lastframe for setting blockSize and
        // gridSize
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);
        jfieldID win_starId = env->GetFieldID(ACFInputParamsCls, "win_star", "I");
        jfieldID hin_starId = env->GetFieldID(ACFInputParamsCls, "hin_star", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID pixbinXId = env->GetFieldID(ACFInputParamsCls, "pixbinX", "I");
        jfieldID pixbinYId = env->GetFieldID(ACFInputParamsCls, "pixbinY", "I");
        jfieldID binningXId = env->GetFieldID(ACFInputParamsCls, "binningX", "I");
        jfieldID binningYId = env->GetFieldID(ACFInputParamsCls, "binningY", "I");
        jfieldID firstframeId = env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId = env->GetFieldID(ACFInputParamsCls, "lastframe", "I");

        jint win_star = env->GetIntField(ACFInputParams, win_starId);
        jint hin_star = env->GetIntField(ACFInputParams, hin_starId);
        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint pixbinX = env->GetIntField(ACFInputParams, pixbinXId);
        jint pixbinY = env->GetIntField(ACFInputParams, pixbinYId);
        jint binningX = env->GetIntField(ACFInputParams, binningXId);
        jint binningY = env->GetIntField(ACFInputParams, binningYId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);

        // Compute frame difference
        int framediff = lastframe - firstframe + 1;

        // Configure CUDA grid and block sizes
        int BLKSIZEXY = 16;
        int max_dim = std::max(w_temp, h_temp);
        int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;

        dim3 blockSizeBin(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSizeBin(GRIDSIZEXY, GRIDSIZEXY, 1);

        // Calculate sizes for device and host memory
        size_t num_input_elements = static_cast<size_t>(win_star) * hin_star * framediff;
        size_t num_output_elements = static_cast<size_t>(w_temp) * h_temp * framediff;

        // Allocate device memory using cuda_utils::CudaDevicePtr
        cuda_utils::CudaDevicePtr<float> d_Cindata(num_input_elements);
        cuda_utils::CudaDevicePtr<float> d_Coutput(num_output_elements);

        // Allocate host memory using std::vector
        std::vector<float> Coutput(num_output_elements);

        // Copy to GPU
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cindata.get(), Cindata, num_input_elements * sizeof(float), cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        calc_binning<<<gridSizeBin, blockSizeBin, 0, stream>>>(d_Cindata.get(), d_Coutput.get(), win_star, hin_star,
                                                               w_temp, h_temp, framediff, pixbinX, pixbinY, binningX,
                                                               binningY);

        // Synchronize and check for errors
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
        CUDA_CHECK_STATUS(cudaGetLastError());

        // copy memory from device to host
        CUDA_CHECK_STATUS(
            cudaMemcpy(Coutput.data(), d_Coutput.get(), num_output_elements * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        CUDA_CHECK_STATUS(cudaDeviceReset());

        // copy values to Java output arrays.
        env->SetFloatArrayRegion(outdata, 0, num_output_elements, Coutput.data());

        // release resources
        env->ReleaseFloatArrayElements(indata, Cindata, 0);
    }
    catch (std::runtime_error &e)
    {
        // Release resources if they were acquired
        if (Cindata != nullptr)
        {
            env->ReleaseFloatArrayElements(indata, Cindata, 0);
        }

        // see: https://www.rgagnon.com/javadetails/java-0323.html
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    isACFmemorySufficient
 * Signature: (LgpufitImFCS/GpufitImFCS/ACFParameters;)Z
 */
jboolean JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_isACFmemorySufficient(JNIEnv *env, jclass cls,
                                                                                             jobject ACFInputParams)
{
    try
    {
        // get parameters that are required for the ACF calculations from the
        // ACFInputParams object
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

        jfieldID widthId = env->GetFieldID(ACFInputParamsCls, "width", "I");
        jfieldID heightId = env->GetFieldID(ACFInputParamsCls, "height", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID firstframeId = env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId = env->GetFieldID(ACFInputParamsCls, "lastframe", "I");
        jfieldID cfXDistanceId = env->GetFieldID(ACFInputParamsCls, "cfXDistance", "I");
        jfieldID cfYDistanceId = env->GetFieldID(ACFInputParamsCls, "cfYDistance", "I");
        //      jfieldID correlatorpId = env->GetFieldID(ACFInputParamsCls,
        //      "correlatorp", "D"); jfieldID correlatorqId =
        //      env->GetFieldID(ACFInputParamsCls, "correlatorq", "D");
        jfieldID frametimeId = env->GetFieldID(ACFInputParamsCls, "frametime", "D");
        jfieldID backgroundId = env->GetFieldID(ACFInputParamsCls, "background", "I");
        jfieldID mtab1Id = env->GetFieldID(ACFInputParamsCls, "mtab1", "D");
        jfieldID mtabchanumminus1Id = env->GetFieldID(ACFInputParamsCls, "mtabchanumminus1", "D");
        jfieldID sampchanumminus1Id = env->GetFieldID(ACFInputParamsCls, "sampchanumminus1", "D");
        jfieldID chanumId = env->GetFieldID(ACFInputParamsCls, "chanum", "I");
        jfieldID isNBcalculationId = env->GetFieldID(ACFInputParamsCls, "isNBcalculation", "Z");
        jfieldID bleachcorr_gpuId = env->GetFieldID(ACFInputParamsCls, "bleachcorr_gpu", "Z");
        jfieldID bleachcorr_orderId = env->GetFieldID(ACFInputParamsCls, "bleachcorr_order", "I");

        jint width = env->GetIntField(ACFInputParams, widthId);
        jint height = env->GetIntField(ACFInputParams, heightId);
        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);
        jint cfXDistance = env->GetIntField(ACFInputParams, cfXDistanceId);
        jint cfYDistance = env->GetIntField(ACFInputParams, cfYDistanceId);
        //      jdouble correlatorpdbl = env->GetDoubleField(ACFInputParams,
        //      correlatorpId); jdouble correlatorqdbl =
        //      env->GetDoubleField(ACFInputParams, correlatorqId);
        jdouble frametime = env->GetDoubleField(ACFInputParams, frametimeId);
        jint background = env->GetIntField(ACFInputParams, backgroundId);
        jdouble mtab1 = env->GetDoubleField(ACFInputParams, mtab1Id); // mtab[1], used to calculate blocknumgpu.
        jdouble mtabchanumminus1 = env->GetDoubleField(ACFInputParams,
                                                       mtabchanumminus1Id); // mtab[chanum-1], used to calculate
                                                                            // pnumgpu[counter_indexarray]
        jdouble sampchanumminus1 = env->GetDoubleField(ACFInputParams,
                                                       sampchanumminus1Id); // samp[chanum-1], used to calculate
                                                                            // pnumgpu[counter_indexarray]
        jint chanum = env->GetIntField(ACFInputParams, chanumId);
        jboolean isNBcalculation = env->GetBooleanField(ACFInputParams, isNBcalculationId);
        jboolean bleachcorr_gpu = env->GetBooleanField(ACFInputParams, bleachcorr_gpuId);
        jint bleachcorr_order = env->GetIntField(ACFInputParams, bleachcorr_orderId);

        // Compute frame difference
        jint framediff = lastframe - firstframe + 1;

        // Size constants
        size_t size_int = sizeof(int);
        size_t size_float = sizeof(float);
        size_t size_double = sizeof(double);

        // Compute sizes of arrays
        size_t size_pixels = static_cast<size_t>(w_temp) * h_temp * framediff * size_float;
        size_t size_Cpixels1 = static_cast<size_t>(width) * height * chanum * size_double;
        size_t size_prod = static_cast<size_t>(framediff) * width * height * size_float;
        size_t size_blockvararray = static_cast<size_t>(chanum) * width * height * size_double;
        size_t size_samp = static_cast<size_t>(chanum) * size_double;
        size_t size_lag = static_cast<size_t>(chanum) * size_int;
        size_t size_bleachcorr = static_cast<size_t>(width) * height * bleachcorr_order * size_double;

        // Total memory required for common parameters
        size_t total_memory_common = size_Cpixels1 + size_prod + size_pixels + size_samp + size_lag + size_bleachcorr;

        // Calculate blocknumgpu
        int blocknumgpu = static_cast<int>(std::floor(std::log(mtab1) / std::log(2.0)) - 2);

        // Memory required for calcacf3
        size_t total_memory_calcacf3 = 0;
        if (!isNBcalculation)
        {
            size_t size_prodnum = static_cast<size_t>(blocknumgpu) * size_double;
            size_t size_upper = static_cast<size_t>(blocknumgpu) * width * height * size_double;
            size_t size_lower = size_upper;
            size_t size_varblock = static_cast<size_t>(blocknumgpu) * width * height * size_double * 3; // varblock0,1,2

            total_memory_calcacf3 = size_prodnum + size_blockvararray + size_upper + size_lower + size_varblock;
        }

        // Memory required for calcacf2
        size_t total_memory_calcacf2 = 0;
        size_t size_prodnumarray = static_cast<size_t>(chanum) * width * height * size_int;
        size_t size_indexarray = static_cast<size_t>(width) * height * size_int;
        size_t size_pnumgpu = size_indexarray;
        size_t size_NB_arrays = 0;

        if (isNBcalculation)
        {
            // Memory for N&B calculations
            size_NB_arrays =
                static_cast<size_t>(width) * height * size_double * 3; // NBmeanGPU, NBmean2GPU, NBcovarianceGPU
        }

        total_memory_calcacf2 =
            size_prodnumarray + size_indexarray + size_blockvararray * 2 + size_pnumgpu + size_NB_arrays;

        // Total memory required
        size_t max_memory_calcacf = std::max(total_memory_calcacf3, total_memory_calcacf2);
        size_t total_memory_required = total_memory_common + max_memory_calcacf;

        // Get available GPU memory
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK_STATUS(cudaMemGetInfo(&free_bytes, &total_bytes));

        // Check if required memory exceeds 90% of free memory
        if (total_memory_required > static_cast<size_t>(free_bytes * 0.9))
        {
            return JNI_FALSE;
        }
        else
        {
            return JNI_TRUE;
        }
    }
    catch (const std::exception &e)
    {
        // Throw Java exception
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
        return JNI_FALSE;
    }
}

void JNICALL Java_fiji_plugin_imaging_1fcs_gpufitImFCS_GpufitImFCS_calcACF(
    JNIEnv *env, jclass cls, jfloatArray pixels, jdoubleArray pixels1, jdoubleArray blockvararray,
    jdoubleArray NBmeanGPU, jdoubleArray NBcovarianceGPU, jdoubleArray blocked1D, jdoubleArray bleachcorr_params,
    jdoubleArray samp, jintArray lag, jobject ACFInputParams)
{
    // Pointers to hold Java array elements
    jfloat *Cpixels = nullptr;
    jdouble *Cbleachcorr_params = nullptr;
    jdouble *Csamp = nullptr;
    jint *Clag = nullptr;

    try
    {
        // Get elements from Java arrays
        Cpixels = env->GetFloatArrayElements(pixels, NULL);
        Cbleachcorr_params = env->GetDoubleArrayElements(bleachcorr_params, NULL);
        Csamp = env->GetDoubleArrayElements(samp, NULL);
        Clag = env->GetIntArrayElements(lag, NULL);

        // Get parameters from ACFInputParams object
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

        // Retrieve field IDs
        jfieldID widthId = env->GetFieldID(ACFInputParamsCls, "width", "I");
        jfieldID heightId = env->GetFieldID(ACFInputParamsCls, "height", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID pixbinXId = env->GetFieldID(ACFInputParamsCls, "pixbinX", "I");
        jfieldID pixbinYId = env->GetFieldID(ACFInputParamsCls, "pixbinY", "I");
        jfieldID firstframeId = env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId = env->GetFieldID(ACFInputParamsCls, "lastframe", "I");
        jfieldID cfXDistanceId = env->GetFieldID(ACFInputParamsCls, "cfXDistance", "I");
        jfieldID cfYDistanceId = env->GetFieldID(ACFInputParamsCls, "cfYDistance", "I");
        jfieldID correlatorpId = env->GetFieldID(ACFInputParamsCls, "correlatorp", "D");
        jfieldID correlatorqId = env->GetFieldID(ACFInputParamsCls, "correlatorq", "D");
        jfieldID frametimeId = env->GetFieldID(ACFInputParamsCls, "frametime", "D");
        jfieldID backgroundId = env->GetFieldID(ACFInputParamsCls, "background", "I");
        jfieldID mtab1Id = env->GetFieldID(ACFInputParamsCls, "mtab1", "D");
        jfieldID mtabchanumminus1Id = env->GetFieldID(ACFInputParamsCls, "mtabchanumminus1", "D");
        jfieldID sampchanumminus1Id = env->GetFieldID(ACFInputParamsCls, "sampchanumminus1", "D");
        jfieldID chanumId = env->GetFieldID(ACFInputParamsCls, "chanum", "I");
        jfieldID isNBcalculationId = env->GetFieldID(ACFInputParamsCls, "isNBcalculation", "Z");
        jfieldID bleachcorr_gpuId = env->GetFieldID(ACFInputParamsCls, "bleachcorr_gpu", "Z");
        jfieldID bleachcorr_orderId = env->GetFieldID(ACFInputParamsCls, "bleachcorr_order", "I");

        // Retrieve field values
        jint width = env->GetIntField(ACFInputParams, widthId);
        jint height = env->GetIntField(ACFInputParams, heightId);
        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint pixbinX = env->GetIntField(ACFInputParams, pixbinXId);
        jint pixbinY = env->GetIntField(ACFInputParams, pixbinYId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);
        jint cfXDistance = env->GetIntField(ACFInputParams, cfXDistanceId);
        jint cfYDistance = env->GetIntField(ACFInputParams, cfYDistanceId);
        jdouble correlatorp = env->GetDoubleField(ACFInputParams, correlatorpId);
        jdouble correlatorq = env->GetDoubleField(ACFInputParams, correlatorqId);
        jdouble frametime = env->GetDoubleField(ACFInputParams, frametimeId);
        jint background = env->GetIntField(ACFInputParams, backgroundId);
        jdouble mtab1 = env->GetDoubleField(ACFInputParams, mtab1Id);
        jdouble mtabchanumminus1 = env->GetDoubleField(ACFInputParams, mtabchanumminus1Id);
        jdouble sampchanumminus1 = env->GetDoubleField(ACFInputParams, sampchanumminus1Id);
        jint chanum = env->GetIntField(ACFInputParams, chanumId);
        jboolean isNBcalculation = env->GetBooleanField(ACFInputParams, isNBcalculationId);
        jboolean bleachcorr_gpu = env->GetBooleanField(ACFInputParams, bleachcorr_gpuId);
        jint bleachcorr_order = env->GetIntField(ACFInputParams, bleachcorr_orderId);

        // Initialize parameters
        int framediff = lastframe - firstframe + 1;
        size_t size_pixels = static_cast<size_t>(w_temp) * h_temp * framediff;
        size_t size1 = static_cast<size_t>(width) * height * chanum;
        size_t size2 = static_cast<size_t>(framediff) * width * height;
        size_t sizeblockvararray = static_cast<size_t>(chanum) * width * height;
        size_t size_bleachcorr_params = static_cast<size_t>(w_temp) * h_temp * bleachcorr_order;

        int blocknumgpu = static_cast<int>(std::floor(std::log(mtab1) / std::log(2.0)) - 2);

        // Configure CUDA grid and block sizes
        int BLKSIZEXY = 16;
        int max_dim = std::max(width, height);
        int GRIDSIZEXY = (max_dim + BLKSIZEXY - 1) / BLKSIZEXY;
        dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

        int max_dim_input = std::max(w_temp, h_temp);
        int GRIDSIZEXY_Input = (max_dim_input + BLKSIZEXY - 1) / BLKSIZEXY;
        dim3 gridSize_Input(GRIDSIZEXY_Input, GRIDSIZEXY_Input, 1);

        // Host arrays using std::vector
        std::vector<double> Cpixels1(size1);
        std::vector<float> prod(size2);
        std::vector<double> Cblocked1D(size1);

        // Host arrays for calcacf3
        std::vector<double> prodnum;
        std::vector<double> upper;
        std::vector<double> lower;
        std::vector<double> varblock0;
        std::vector<double> varblock1;
        std::vector<double> varblock2;

        if (!isNBcalculation)
        {
            size_t size_prodnum = static_cast<size_t>(blocknumgpu);
            size_t size_upper_lower = static_cast<size_t>(blocknumgpu) * width * height;
            size_t size_varblock = static_cast<size_t>(blocknumgpu) * width * height;

            prodnum.resize(size_prodnum, 1.0);
            upper.resize(size_upper_lower, 0.0);
            lower.resize(size_upper_lower, 0.0);
            varblock0.resize(size_varblock, 0.0);
            varblock1.resize(size_varblock, 0.0);
            varblock2.resize(size_varblock, 0.0);
        }

        // Host arrays for calcacf2
        std::vector<int> prodnumarray(chanum * width * height);
        std::vector<int> indexarray(width * height);
        std::vector<double> Cblockvararray(sizeblockvararray);
        std::vector<double> blocksdarray(sizeblockvararray);
        std::vector<int> pnumgpu(width * height);

        // Host arrays for N & B calculations
        std::vector<double> CNBmeanGPU;
        std::vector<double> CNBcovarianceGPU;

        if (isNBcalculation)
        {
            CNBmeanGPU.resize(width * height, 0.0);
            CNBcovarianceGPU.resize(width * height, 0.0);
        }

        // Create CUDA stream
        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        // Device arrays using cuda_utils::CudaDevicePtr
        cuda_utils::CudaDevicePtr<float> d_Cpixels(size_pixels);
        cuda_utils::CudaDevicePtr<double> d_Cpixels1(size1);
        cuda_utils::CudaDevicePtr<double> d_Cbleachcorr_params(size_bleachcorr_params);
        cuda_utils::CudaDevicePtr<double> d_Csamp(chanum);
        cuda_utils::CudaDevicePtr<int> d_Clag(chanum);
        cuda_utils::CudaDevicePtr<float> d_prod(size2);

        // Device arrays for calcacf3
        cuda_utils::CudaDevicePtr<double> d_prodnum;
        cuda_utils::CudaDevicePtr<double> d_upper;
        cuda_utils::CudaDevicePtr<double> d_lower;
        cuda_utils::CudaDevicePtr<double> d_varblock0;
        cuda_utils::CudaDevicePtr<double> d_varblock1;
        cuda_utils::CudaDevicePtr<double> d_varblock2;

        if (!isNBcalculation)
        {
            d_prodnum = cuda_utils::CudaDevicePtr<double>(prodnum.size());
            d_upper = cuda_utils::CudaDevicePtr<double>(upper.size());
            d_lower = cuda_utils::CudaDevicePtr<double>(lower.size());
            d_varblock0 = cuda_utils::CudaDevicePtr<double>(varblock0.size());
            d_varblock1 = cuda_utils::CudaDevicePtr<double>(varblock1.size());
            d_varblock2 = cuda_utils::CudaDevicePtr<double>(varblock2.size());
        }

        // Copy data to device
        CUDA_CHECK_STATUS(
            cudaMemcpyAsync(d_Cpixels.get(), Cpixels, size_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));
        // Initialize d_Cpixels1 if needed
        CUDA_CHECK_STATUS(
            cudaMemcpyAsync(d_Cpixels1.get(), Cpixels1.data(), size1 * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Cbleachcorr_params.get(), Cbleachcorr_params,
                                          size_bleachcorr_params * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Clag.get(), Clag, chanum * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(
            cudaMemcpyAsync(d_prod.get(), prod.data(), size2 * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Copy data for calcacf3
        if (!isNBcalculation)
        {
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_prodnum.get(), prodnum.data(), prodnum.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_upper.get(), upper.data(), upper.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_lower.get(), lower.data(), lower.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_varblock0.get(), varblock0.data(), varblock0.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_varblock1.get(), varblock1.data(), varblock1.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_varblock2.get(), varblock2.data(), varblock2.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
        }

        // Synchronize stream to ensure data is copied
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

        // Run kernels for calcacf3
        if (!isNBcalculation)
        {
            if (bleachcorr_gpu)
            {
                bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(d_Cpixels.get(), w_temp, h_temp, framediff,
                                                                           bleachcorr_order, frametime,
                                                                           d_Cbleachcorr_params.get());
                CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
                CUDA_CHECK_STATUS(cudaGetLastError());
            }

            calcacf3<<<gridSize, blockSize, 0, stream>>>(
                d_Cpixels.get(), cfXDistance, cfYDistance, 1, width, height, w_temp, h_temp, pixbinX, pixbinY,
                framediff, static_cast<int>(correlatorq), frametime, d_Cpixels1.get(), d_prod.get(),
                d_prodnum.get(), d_upper.get(), d_lower.get(), d_varblock0.get(), d_varblock1.get(),
                d_varblock2.get(), d_Clag.get());

            CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
            CUDA_CHECK_STATUS(cudaGetLastError());

            // Copy results back to host
            CUDA_CHECK_STATUS(cudaMemcpyAsync(Cpixels1.data(), d_Cpixels1.get(), size1 * sizeof(double),
                                              cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

            // Copy Cpixels1 to Cblocked1D
            std::copy(Cpixels1.begin(), Cpixels1.end(), Cblocked1D.begin());

            // Initialize indexarray and pnumgpu
            size_t counter_indexarray = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    indexarray[counter_indexarray] = static_cast<int>(Cpixels1[idx]);

                    // Minimum number of products
                    double tempval = indexarray[counter_indexarray] - std::log2(sampchanumminus1);
                    if (tempval < 0)
                    {
                        tempval = 0;
                    }
                    pnumgpu[counter_indexarray] =
                        static_cast<int>(std::floor(mtabchanumminus1 / std::pow(2.0, tempval)));
                    counter_indexarray++;
                }
            }
        }

        // Re-copy Cpixels to device if modified
        CUDA_CHECK_STATUS(
            cudaMemcpyAsync(d_Cpixels.get(), Cpixels, size_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

        // Device arrays for calcacf2
        cuda_utils::CudaDevicePtr<int> d_prodnumarray(chanum * width * height);
        cuda_utils::CudaDevicePtr<int> d_indexarray(width * height);
        cuda_utils::CudaDevicePtr<double> d_Cblockvararray(sizeblockvararray);
        cuda_utils::CudaDevicePtr<double> d_blocksdarray(sizeblockvararray);
        cuda_utils::CudaDevicePtr<int> d_pnumgpu(width * height);

        // Device arrays for N & B calculations
        cuda_utils::CudaDevicePtr<double> d_NBmeanGPU;
        cuda_utils::CudaDevicePtr<double> d_NBcovarianceGPU;

        if (isNBcalculation)
        {
            d_NBmeanGPU = cuda_utils::CudaDevicePtr<double>(width * height);
            d_NBcovarianceGPU = cuda_utils::CudaDevicePtr<double>(width * height);

            // Copy N & B arrays to device
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_NBmeanGPU.get(), CNBmeanGPU.data(), CNBmeanGPU.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(d_NBcovarianceGPU.get(), CNBcovarianceGPU.data(),
                                              CNBcovarianceGPU.size() * sizeof(double), cudaMemcpyHostToDevice,
                                              stream));
        }

        // Copy data for calcacf2
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_prodnumarray.get(), prodnumarray.data(), prodnumarray.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_indexarray.get(), indexarray.data(), indexarray.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_Cblockvararray.get(), Cblockvararray.data(),
                                          Cblockvararray.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_blocksdarray.get(), blocksdarray.data(),
                                          blocksdarray.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaMemcpyAsync(d_pnumgpu.get(), pnumgpu.data(), pnumgpu.size() * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

        // Run kernels for calcacf2
        if (bleachcorr_gpu)
        {
            bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(
                d_Cpixels.get(), w_temp, h_temp, framediff, bleachcorr_order, frametime, d_Cbleachcorr_params.get());
            CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
            CUDA_CHECK_STATUS(cudaGetLastError());
        }

        int numbin = framediff;
        int currentIncrement = 1;
        int ctbin = 0;

        int calcacf2_x_start = isNBcalculation ? 1 : 0;
        int calcacf2_x_end = isNBcalculation ? 2 : chanum;

        for (int x = calcacf2_x_start; x < calcacf2_x_end; x++)
        {
            bool runthis = false;
            if (currentIncrement != Csamp[x])
            {
                numbin = static_cast<int>(std::floor(static_cast<double>(numbin) / 2.0));
                currentIncrement = static_cast<int>(Csamp[x]);
                ctbin++;
                runthis = true;
            }

            if (runthis)
            {
                calcacf2a<<<gridSize_Input, blockSize, 0, stream>>>(d_Cpixels.get(), w_temp, h_temp, numbin);
                CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
                CUDA_CHECK_STATUS(cudaGetLastError());
            }

            if (isNBcalculation)
            {
                // Launch calcacf2b_NB
                calcacf2b_NB<<<gridSize, blockSize>>>(d_Cpixels.get(), cfXDistance, cfYDistance, width, height, w_temp,
                                                      h_temp, pixbinX, pixbinY, d_prod.get(), d_Clag.get(),
                                                      d_prodnumarray.get(), d_NBmeanGPU.get(), d_NBcovarianceGPU.get(),
                                                      x, numbin, currentIncrement);
            }
            else
            {
                // Launch calcacf2b_ACF
                calcacf2b_ACF<<<gridSize, blockSize>>>(
                    d_Cpixels.get(), cfXDistance, cfYDistance, width, height, w_temp, h_temp, pixbinX, pixbinY,
                    d_Cpixels1.get(), d_prod.get(), d_Clag.get(), d_prodnumarray.get(), d_indexarray.get(),
                    d_Cblockvararray.get(), d_blocksdarray.get(), d_pnumgpu.get(), x, numbin, currentIncrement, ctbin);
            }

            CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));
            CUDA_CHECK_STATUS(cudaGetLastError());
        }

        // Copy results back to host
        if (isNBcalculation)
        {
            CUDA_CHECK_STATUS(cudaMemcpyAsync(CNBmeanGPU.data(), d_NBmeanGPU.get(), CNBmeanGPU.size() * sizeof(double),
                                              cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(CNBcovarianceGPU.data(), d_NBcovarianceGPU.get(),
                                              CNBcovarianceGPU.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                              stream));
        }
        else
        {
            CUDA_CHECK_STATUS(cudaMemcpyAsync(Cpixels1.data(), d_Cpixels1.get(), size1 * sizeof(double),
                                              cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_STATUS(cudaMemcpyAsync(Cblockvararray.data(), d_Cblockvararray.get(),
                                              Cblockvararray.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK_STATUS(cudaStreamSynchronize(stream));

        // Destroy CUDA stream
        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        // Release Java array elements
        env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
        env->ReleaseDoubleArrayElements(bleachcorr_params, Cbleachcorr_params, 0);
        env->ReleaseDoubleArrayElements(samp, Csamp, 0);
        env->ReleaseIntArrayElements(lag, Clag, 0);

        // Copy results back to Java arrays
        env->SetDoubleArrayRegion(pixels1, 0, size1, Cpixels1.data());
        env->SetDoubleArrayRegion(blockvararray, 0, Cblockvararray.size(), Cblockvararray.data());
        env->SetDoubleArrayRegion(blocked1D, 0, size1, Cblocked1D.data());

        if (isNBcalculation)
        {
            env->SetDoubleArrayRegion(NBmeanGPU, 0, CNBmeanGPU.size(), CNBmeanGPU.data());
            env->SetDoubleArrayRegion(NBcovarianceGPU, 0, CNBcovarianceGPU.size(), CNBcovarianceGPU.data());
        }
    }
    catch (const std::exception &e)
    {
        // Release Java array elements if they were acquired
        if (Cpixels != nullptr)
        {
            env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
        }
        if (Cbleachcorr_params != nullptr)
        {
            env->ReleaseDoubleArrayElements(bleachcorr_params, Cbleachcorr_params, 0);
        }
        if (Csamp != nullptr)
        {
            env->ReleaseDoubleArrayElements(samp, Csamp, 0);
        }
        if (Clag != nullptr)
        {
            env->ReleaseIntArrayElements(lag, Clag, 0);
        }

        // Throw Java exception
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }
}