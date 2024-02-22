#include <cuda_runtime.h>

#include "definitions.h"
#include "gpufit.h"
#include "gpufitImFCS_GpufitImFCS.h"

/*-------------------------------------------------------------------------------------------------------
* Calculate binning START
-------------------------------------------------------------------------------------------------------*/
__global__ void calc_binning(float *data, float *data1, int win_star,
                             int hin_star, int w_temp, int h_temp,
                             int framediff, int pixbinX, int pixbinY,
                             int binningX, int binningY)
{
    // this function performs binning of spatial data.

    // NOTE: In the case overlap is OFF, we sill bin for every pixel, one pixel
    // at a time. This allows us to use cfXDistance and cfYDistance directly
    // instead of translating these distances, which will be difficult.

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();

    float sum = 0.0;

    if ((idx < w_temp) && (idy < h_temp))
    {
        for (int t = 0; t < framediff; t++)
        {
            sum = 0.0;
            for (int i = 0; i < binningX; i++)
            {
                for (int j = 0; j < binningY; j++)
                {
                    sum += data[t * win_star * hin_star + (idy + j) * win_star
                                + (idx + i)];
                } // for j
            } // for i

            data1[t * w_temp * h_temp + idy * w_temp + idx] = sum;

        } // for t
    } // if
}

/*-------------------------------------------------------------------------------------------------------
* Calculate binning END
-------------------------------------------------------------------------------------------------------*/

__global__ void calc_data_bleach_correction(float *data, float *data1,
                                            int width, int height, int nopit,
                                            int ave)
{
    // function is an averaging step in temporal dimension for every ave number
    // of points, prior to performing bleach correction fitting.

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y;
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
jint JNICALL Java_gpufitImFCS_GpufitImFCS_fit(
    JNIEnv *env, jclass cls, jint number_fits, jint number_points,
    jobject data_buffer, jobject weights_buffer, jint model_id,
    jobject initial_parameter_buffer, jfloat tolerance,
    jint max_number_iterations, jint num_valid_coefs,
    jobject paramters_to_fit_buffer, jint estimator_id, jint user_info_size,
    jobject user_info_buffer, jobject output_parameters_buffer,
    jobject output_states_buffer, jobject output_chi_squares_buffer,
    jobject output_number_iterations_buffer)
{
    // get pointer to buffers
    REAL *data = (REAL *)buffer_address(env, data_buffer);
    REAL *weights = (REAL *)buffer_address(env, weights_buffer);
    REAL *initial_parameters =
        (REAL *)buffer_address(env, initial_parameter_buffer);
    int *parameters_to_fit =
        (int *)buffer_address(env, paramters_to_fit_buffer);
    char *user_info = (char *)buffer_address(env, user_info_buffer);
    REAL *output_parameters =
        (REAL *)buffer_address(env, output_parameters_buffer);
    int *output_states = (int *)buffer_address(env, output_states_buffer);
    REAL *output_chi_squares =
        (REAL *)buffer_address(env, output_chi_squares_buffer);
    int *output_number_iterations =
        (int *)buffer_address(env, output_number_iterations_buffer);

    // call to gpufit
    // NOTE: Added num_valid_coefs
    int status = gpufit(
        number_fits, number_points, data, weights, model_id, initial_parameters,
        tolerance, max_number_iterations, num_valid_coefs, parameters_to_fit,
        estimator_id, user_info_size, user_info, output_parameters,
        output_states, output_chi_squares, output_number_iterations);

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
jstring JNICALL Java_gpufitImFCS_GpufitImFCS_getLastError(JNIEnv *env,
                                                          jclass cls)
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
jboolean JNICALL Java_gpufitImFCS_GpufitImFCS_isCudaAvailableInt(JNIEnv *env,
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
jintArray JNICALL
Java_gpufitImFCS_GpufitImFCS_getCudaVersionAsArray(JNIEnv *env, jclass cls)
{
    int runtime_version, driver_version;
    if (gpufit_get_cuda_version(&runtime_version, &driver_version)
        == ReturnState::OK)
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
void JNICALL Java_gpufitImFCS_GpufitImFCS_resetGPU(JNIEnv *env, jclass cls)
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
void JNICALL Java_gpufitImFCS_GpufitImFCS_calcDataBleachCorrection(
    JNIEnv *env, jclass cls, jfloatArray pixels, jfloatArray outdata,
    jobject ACFInputParams)
{
    size_t SIZEFLOAT = sizeof(float);

    // input arrays required for calculations.
    jfloat *Cpixels;
    float *d_Cpixels;

    // output data
    float *Coutput;
    float *d_Coutput;

    try
    {
        Cpixels = env->GetFloatArrayElements(pixels, NULL);

        // get parameters from the ACFInputParams object
        // we need width, height, cfXDistancegpu, cfYDistancegpu, nopit, ave
        // we also need firstframe and lastframe for setting blockSize and
        // gridSize
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID firstframeId =
            env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId =
            env->GetFieldID(ACFInputParamsCls, "lastframe", "I");
        jfieldID nopitId = env->GetFieldID(ACFInputParamsCls, "nopit", "I");
        jfieldID aveId = env->GetFieldID(ACFInputParamsCls, "ave", "I");

        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);
        jint nopit = env->GetIntField(ACFInputParams, nopitId);
        jint ave = env->GetIntField(ACFInputParams, aveId);

        // blockSize and gridSize
        int framediff = lastframe - firstframe + 1;
        int BLKSIZEXY = 16;
        int a = (w_temp > h_temp) ? w_temp : h_temp;
        int GRIDSIZEXY = (a + BLKSIZEXY - 1) / BLKSIZEXY;

        dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

        // Allocate memory on GPU
        size_t size = w_temp * h_temp * framediff * SIZEFLOAT;
        cudaMalloc((void **)&d_Cpixels, size);

        // Allocate memory for Coutput and d_Coutput
        unsigned int sizeA = w_temp * h_temp * nopit;
        size_t size1 = sizeA * SIZEFLOAT;

        Coutput = (float *)malloc(size1);
        cudaMalloc((void **)&d_Coutput, size1);

        // Copy to GPU
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cpixels, Cpixels, size, cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        calc_data_bleach_correction<<<gridSize, blockSize, 0, stream>>>(
            d_Cpixels, d_Coutput, w_temp, h_temp, nopit, ave);

        cudaDeviceSynchronize();
        CUDA_CHECK_STATUS(cudaGetLastError());

        // copy memory from device to host
        CUDA_CHECK_STATUS(
            cudaMemcpy(Coutput, d_Coutput, size1, cudaMemcpyDeviceToHost));

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        // CUDA release memory
        cudaFree(d_Cpixels);
        cudaFree(d_Coutput);

        cudaDeviceReset();

        // copy values to Java output arrays.
        env->SetFloatArrayRegion(outdata, 0, sizeA, Coutput);

        // free pointers
        free(Coutput);

        // release resources
        env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
    }
    catch (std::runtime_error &e)
    {
        // CUDA release memory
        cudaFree(d_Cpixels);
        cudaFree(d_Coutput);

        // free pointers
        free(Coutput);

        // release resources
        env->ReleaseFloatArrayElements(pixels, Cpixels, 0);

        // see: https://www.rgagnon.com/javadetails/java-0323.html
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }

    return;
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    isBinningMemorySufficient
 * Signature: (LgpufitImFCS/GpufitImFCS/ACFParameters;)Z
 */
jboolean JNICALL Java_gpufitImFCS_GpufitImFCS_isBinningMemorySufficient(
    JNIEnv *env, jclass cls, jobject ACFInputParams)
{
    try
    {
        unsigned int SIZEFLOAT = sizeof(float);

        // get parameters from the ACFInputParams object
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);
        jfieldID win_starId =
            env->GetFieldID(ACFInputParamsCls, "win_star", "I");
        jfieldID hin_starId =
            env->GetFieldID(ACFInputParamsCls, "hin_star", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID firstframeId =
            env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId =
            env->GetFieldID(ACFInputParamsCls, "lastframe", "I");

        jint win_star = env->GetIntField(ACFInputParams, win_starId);
        jint hin_star = env->GetIntField(ACFInputParams, hin_starId);
        jint w_temp = env->GetIntField(ACFInputParams, w_tempId);
        jint h_temp = env->GetIntField(ACFInputParams, h_tempId);
        jint firstframe = env->GetIntField(ACFInputParams, firstframeId);
        jint lastframe = env->GetIntField(ACFInputParams, lastframeId);

        // sanity check if memory on GPU is sufficient for binning
        std::size_t this_free_bytes;
        std::size_t this_total_bytes;
        int framediff = lastframe - firstframe + 1;
        double maxmemory = (double)(win_star * hin_star + w_temp * h_temp)
            * framediff * SIZEFLOAT;
        CUDA_CHECK_STATUS(cudaMemGetInfo(&this_free_bytes, &this_total_bytes));

        return (maxmemory > double(this_free_bytes) * 0.9) ? JNI_FALSE
                                                           : JNI_TRUE;
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
void JNICALL Java_gpufitImFCS_GpufitImFCS_calcBinning(JNIEnv *env, jclass cls,
                                                      jfloatArray indata,
                                                      jfloatArray outdata,
                                                      jobject ACFInputParams)
{
    size_t SIZEFLOAT = sizeof(float);

    // input arrays required for calculations.
    jfloat *Cindata;
    float *d_Cindata;

    // output data
    float *Coutput;
    float *d_Coutput;

    try
    {
        Cindata = env->GetFloatArrayElements(indata, NULL);

        // get parameters from the ACFInputParams object
        // we need w_temp, h_temp, binningX, binningY
        // we also need firstframe and lastframe for setting blockSize and
        // gridSize
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);
        jfieldID win_starId =
            env->GetFieldID(ACFInputParamsCls, "win_star", "I");
        jfieldID hin_starId =
            env->GetFieldID(ACFInputParamsCls, "hin_star", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID pixbinXId = env->GetFieldID(ACFInputParamsCls, "pixbinX", "I");
        jfieldID pixbinYId = env->GetFieldID(ACFInputParamsCls, "pixbinY", "I");
        jfieldID binningXId =
            env->GetFieldID(ACFInputParamsCls, "binningX", "I");
        jfieldID binningYId =
            env->GetFieldID(ACFInputParamsCls, "binningY", "I");
        jfieldID firstframeId =
            env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId =
            env->GetFieldID(ACFInputParamsCls, "lastframe", "I");

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

        // blockSize and gridSize
        int framediff = lastframe - firstframe + 1;
        int BLKSIZEXY = 16;
        int a = (w_temp > h_temp) ? w_temp : h_temp;
        int GRIDSIZEXY = (a + BLKSIZEXY - 1) / BLKSIZEXY;

        dim3 blockSizeBin(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSizeBin(GRIDSIZEXY, GRIDSIZEXY, 1);

        // Allocate memory on GPU
        size_t size = win_star * hin_star * framediff * SIZEFLOAT;
        cudaMalloc((void **)&d_Cindata, size);

        // Allocate memory for Coutput and d_Coutput
        unsigned int sizeA = w_temp * h_temp * framediff;
        size_t size1 = sizeA * SIZEFLOAT;
        Coutput = (float *)malloc(size1);
        cudaMalloc((void **)&d_Coutput, size1);

        // Copy to GPU
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cindata, Cindata, size, cudaMemcpyHostToDevice));

        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        calc_binning<<<gridSizeBin, blockSizeBin, 0, stream>>>(
            d_Cindata, d_Coutput, win_star, hin_star, w_temp, h_temp, framediff,
            pixbinX, pixbinY, binningX, binningY);

        cudaDeviceSynchronize();
        CUDA_CHECK_STATUS(cudaGetLastError());

        // copy memory from device to host
        CUDA_CHECK_STATUS(
            cudaMemcpy(Coutput, d_Coutput, size1, cudaMemcpyDeviceToHost));

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        // CUDA release memory
        cudaFree(d_Cindata);
        cudaFree(d_Coutput);

        cudaDeviceReset();

        // copy values to Java output arrays.
        env->SetFloatArrayRegion(outdata, 0, sizeA, Coutput);

        // free pointers
        free(Coutput);

        // release resources
        env->ReleaseFloatArrayElements(indata, Cindata, 0);
    }
    catch (std::runtime_error &e)
    {
        // CUDA release memory
        cudaFree(d_Cindata);
        cudaFree(d_Coutput);

        // free pointers
        free(Coutput);

        // release resources
        env->ReleaseFloatArrayElements(indata, Cindata, 0);

        // see: https://www.rgagnon.com/javadetails/java-0323.html
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
    }

    return;
}

/*
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    isACFmemorySufficient
 * Signature: (LgpufitImFCS/GpufitImFCS/ACFParameters;)Z
 */
jboolean JNICALL Java_gpufitImFCS_GpufitImFCS_isACFmemorySufficient(
    JNIEnv *env, jclass cls, jobject ACFInputParams)
{
    try
    {
        unsigned int SIZEINT = sizeof(int);
        unsigned int SIZEFLOAT = sizeof(float);
        unsigned int SIZEDOUBLE = sizeof(double);

        double totalmemoryAll = 0.0;
        double totalmemoryCalc3 = 0.0;
        double totalmemoryCalc2 = 0.0;

        // get parameters that are required for the ACF calculations from the
        // ACFInputParams object
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

        jfieldID widthId = env->GetFieldID(ACFInputParamsCls, "width", "I");
        jfieldID heightId = env->GetFieldID(ACFInputParamsCls, "height", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID firstframeId =
            env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId =
            env->GetFieldID(ACFInputParamsCls, "lastframe", "I");
        jfieldID cfXDistanceId =
            env->GetFieldID(ACFInputParamsCls, "cfXDistance", "I");
        jfieldID cfYDistanceId =
            env->GetFieldID(ACFInputParamsCls, "cfYDistance", "I");
        //      jfieldID correlatorpId = env->GetFieldID(ACFInputParamsCls,
        //      "correlatorp", "D"); jfieldID correlatorqId =
        //      env->GetFieldID(ACFInputParamsCls, "correlatorq", "D");
        jfieldID frametimeId =
            env->GetFieldID(ACFInputParamsCls, "frametime", "D");
        jfieldID backgroundId =
            env->GetFieldID(ACFInputParamsCls, "background", "I");
        jfieldID mtab1Id = env->GetFieldID(ACFInputParamsCls, "mtab1", "D");
        jfieldID mtabchanumminus1Id =
            env->GetFieldID(ACFInputParamsCls, "mtabchanumminus1", "D");
        jfieldID sampchanumminus1Id =
            env->GetFieldID(ACFInputParamsCls, "sampchanumminus1", "D");
        jfieldID chanumId = env->GetFieldID(ACFInputParamsCls, "chanum", "I");
        jfieldID isNBcalculationId =
            env->GetFieldID(ACFInputParamsCls, "isNBcalculation", "Z");
        jfieldID bleachcorr_gpuId =
            env->GetFieldID(ACFInputParamsCls, "bleachcorr_gpu", "Z");
        jfieldID bleachcorr_orderId =
            env->GetFieldID(ACFInputParamsCls, "bleachcorr_order", "I");

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
        jdouble mtab1 = env->GetDoubleField(
            ACFInputParams, mtab1Id); // mtab[1], used to calculate blocknumgpu.
        jdouble mtabchanumminus1 = env->GetDoubleField(
            ACFInputParams,
            mtabchanumminus1Id); // mtab[chanum-1], used to calculate
                                 // pnumgpu[counter_indexarray]
        jdouble sampchanumminus1 = env->GetDoubleField(
            ACFInputParams,
            sampchanumminus1Id); // samp[chanum-1], used to calculate
                                 // pnumgpu[counter_indexarray]
        jint chanum = env->GetIntField(ACFInputParams, chanumId);
        jboolean isNBcalculation =
            env->GetBooleanField(ACFInputParams, isNBcalculationId);
        jboolean bleachcorr_gpu =
            env->GetBooleanField(ACFInputParams, bleachcorr_gpuId);
        jint bleachcorr_order =
            env->GetIntField(ACFInputParams, bleachcorr_orderId);

        // initialize parameters
        //      int correlatorp = (int) correlatorpdbl;
        //      int correlatorq = (int) correlatorqdbl;
        int framediff = lastframe - firstframe + 1;
        unsigned long size = w_temp * h_temp * framediff * SIZEFLOAT;
        unsigned long size1 = width * height * chanum * SIZEDOUBLE;
        unsigned long size2 = framediff * width * height * SIZEFLOAT;
        unsigned long sizeblockvararray = chanum * width * height * SIZEDOUBLE;

        int blocknumgpu = (int)(floor(log(mtab1) / log(2)) - 2);

        // dynamic memory allocation and/or initialization
        //------------------ common parameters ---------------------------
        totalmemoryAll = totalmemoryAll + (double)size1; // Cpixels1
        totalmemoryAll = totalmemoryAll + (double)size2; // prod
        totalmemoryAll = totalmemoryAll + (double)size; // pixels
        totalmemoryAll = totalmemoryAll + (double)(chanum * SIZEDOUBLE); // samp
        totalmemoryAll = totalmemoryAll + (double)(chanum * SIZEINT); // lag
        totalmemoryAll = totalmemoryAll
            + (double)width * height * bleachcorr_order * SIZEDOUBLE;
        // totalmemoryAll = totalmemoryAll + (double) size1; // Cblocked1D
        // copies Cpixels1 array after calcacf3 calculation and not required GPU
        // memory.

        //------------------ calcacf3 ---------------------------
        if (!isNBcalculation)
        {
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)(blocknumgpu * SIZEDOUBLE); // prodnum
            totalmemoryCalc3 =
                totalmemoryCalc3 + (double)sizeblockvararray; // blocksd
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)(blocknumgpu * width * height * SIZEDOUBLE); // upper
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)(blocknumgpu * width * height * SIZEDOUBLE); // lower
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)((blocknumgpu - 1) * width * height * SIZEINT); // crt
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)((blocknumgpu - 2) * width * height
                           * SIZEINT); // cr12
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)((blocknumgpu - 2) * width * height * SIZEINT); // cr3
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)((blocknumgpu - 1) * width * height
                           * SIZEINT); // diffpos
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)(blocknumgpu * width * height
                           * SIZEDOUBLE); // varblock0
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)(blocknumgpu * width * height
                           * SIZEDOUBLE); // varblock1
            totalmemoryCalc3 = totalmemoryCalc3
                + (double)(blocknumgpu * width * height
                           * SIZEDOUBLE); // varblock2
        }

        //------------------ calcacf2 ---------------------------
        totalmemoryCalc2 = totalmemoryCalc2
            + (double)(chanum * width * height * SIZEINT); // prodnumarray
        totalmemoryCalc2 =
            totalmemoryCalc2 + (double)(width * height * SIZEINT); // indexarray
        totalmemoryCalc2 =
            totalmemoryCalc2 + (double)sizeblockvararray; // Cblockvararray
        totalmemoryCalc2 =
            totalmemoryCalc2 + (double)sizeblockvararray; // blocksdarray
        totalmemoryCalc2 =
            totalmemoryCalc2 + (double)(width * height * SIZEINT); // pnumgpu

        //------------------ calculation of N & B in calcacf2
        //--------------------------
        if (isNBcalculation)
        {
            totalmemoryCalc2 = totalmemoryCalc2
                + (double)(width * height * SIZEDOUBLE
                           * 3); // NBmeanGPU, NBmean2GPU, NBcovarianceGPU
        }

        // sanity check if memory on GPU is sufficient for calcacf3 and calcacf2
        // NOTE calcacf3 will run first. Memory of parameters related to
        // calcacf3 only will be released after completion of calcacf3.
        std::size_t this_free_bytes;
        std::size_t this_total_bytes;
        double maxmemory = (totalmemoryCalc3 > totalmemoryCalc2)
            ? totalmemoryCalc3
            : totalmemoryCalc2;
        maxmemory = maxmemory + totalmemoryAll;
        CUDA_CHECK_STATUS(cudaMemGetInfo(&this_free_bytes, &this_total_bytes));

        return (maxmemory > double(this_free_bytes) * 0.9) ? JNI_FALSE
                                                           : JNI_TRUE;
    }
    catch (std::runtime_error &e)
    {
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());
        return JNI_FALSE;
    }
}

/* ------------------------------------------
AUTOCORRELATION SINGLE DIMENSION ARRAY CALCULATION START
NOTE: SEE JCudaImageJExampleKernelcalcacf2.cu
------------------------------------------ */
__global__ void bleachcorrection(float *data, int w_temp, int h_temp, int d,
                                 int bleachcorr_order, double frametimegpu,
                                 double *bleachcorr_params)
{
    // function performs polynomial bleach correction given polynomial order and
    // coefficients. It is done prior to calcacf3, calcacf2a and calcacf2b.

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();
    if ((idx < w_temp) && (idy < h_temp))
    {
        for (int i = 0; i < d; i++)
        {
            float corfunc = 0;
            for (int ii = 0; ii < bleachcorr_order; ii++)
            {
                corfunc +=
                    bleachcorr_params[(idy * w_temp + idx) * bleachcorr_order
                                      + ii]
                    * powf((float)frametimegpu * (i + 0.5), (float)ii);
            } // for ii

            float res0 =
                bleachcorr_params[(idy * w_temp + idx) * bleachcorr_order];

            data[i * w_temp * h_temp + idy * w_temp + idx] =
                data[i * w_temp * h_temp + idy * w_temp + idx]
                    / sqrtf(corfunc / res0)
                + res0 * (1 - sqrtf(corfunc / res0));
            __syncthreads();
        } // for i
    } // if ((idx < w_temp) && (idy < h_temp))
} // bleachcorrection function

__device__ unsigned int countx = 0;
__device__ unsigned int county = 0;
__shared__ bool isLastBlockDone;
__global__ void calcacf2a(float *data, int w_temp, int h_temp, int numbin)
{
    // function calculates the arrays according to different time bins in
    // different parts of the correlation function

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y;

    __syncthreads();
    if ((idx < w_temp) && (idy < h_temp))
    {
        // And correct the number Of actual data points accordingly
        for (int y = 0; y < numbin; y++)
        { // if yes, bin the data according to the width of the current channel
            data[y * w_temp * h_temp + idy * w_temp + idx] =
                data[2 * y * w_temp * h_temp + idy * w_temp + idx]
                + data[(2 * y + 1) * w_temp * h_temp + idy * w_temp + idx];
            __syncthreads();
        } // for int y = 0
    }
}

__global__ void calcacf2b(float *data, int cfXDistancegpu, int cfYDistancegpu,
                          int w, int h, int w_temp, int h_temp, int pixbinX,
                          int pixbinY, double *data1, float *prod, int *laggpu,
                          int *prodnumarray, int *indexarray,
                          double *blockvararray, double *sdarray, int *pnumgpu,
                          int x, int numbin, int currentIncrement, int ctbin,
                          bool isNBcalculation, double *NBmeanGPU,
                          double *NBcovarianceGPU)
{
    // function calculates the value of the auto or cross-correlation at every
    // lag time. This function also performs the G1 analysis in N and B
    // calculation.

    int del; // delay Or correlation time expressed In lags
    double sumprod = 0.0; // sum of all intensity products; divide by num to get
                          // the average <i(n)i(n+del)>

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y;
    double temp1 = 0.0, temp2 = 0.0;

    __syncthreads();
    if ((idx < w) && (idy < h))
    {
        del = laggpu[x] / currentIncrement;
        prodnumarray[x * w * h + idy * w + idx] = numbin - del;

        temp1 = 0.0;
        temp2 = 0.0;

        for (int y = 0; y < prodnumarray[x * w * h + idy * w + idx]; y++)
        { // calculate the ...
            temp1 += data[y * w_temp * h_temp + idy * pixbinY * w_temp
                          + idx * pixbinX];
            temp2 += data[(y + del) * w_temp * h_temp
                          + (idy * pixbinY + cfYDistancegpu) * w_temp
                          + (idx * pixbinX + cfXDistancegpu)];
        }

        temp1 /= prodnumarray[x * w * h + idy * w
                              + idx]; // calculate average of direct and delayed
                                      // monitor, i.e. the average intensity
                                      // <n(0)> and <n(tau)>
        temp2 /= prodnumarray[x * w * h + idy * w + idx];
        sumprod = 0.0;

        for (int y = 0; y < prodnumarray[x * w * h + idy * w + idx]; y++)
        { // calculate the correlation
            if (isNBcalculation)
            {
                prod[y * w * h + idy * w + idx] =
                    data[y * w_temp * h_temp + idy * pixbinY * w_temp
                         + idx * pixbinX]
                    * data[(y + del) * w_temp * h_temp
                           + (idy * pixbinY + cfYDistancegpu) * w_temp
                           + (idx * pixbinX + cfXDistancegpu)];
            }
            else
            {
                prod[y * w * h + idy * w + idx] =
                    data[y * w_temp * h_temp + idy * pixbinY * w_temp
                         + idx * pixbinX]
                        * data[(y + del) * w_temp * h_temp
                               + (idy * pixbinY + cfYDistancegpu) * w_temp
                               + (idx * pixbinX + cfXDistancegpu)]
                    - temp2
                        * data[y * w_temp * h_temp + idy * pixbinY * w_temp
                               + idx * pixbinX]
                    - temp1
                        * data[(y + del) * w_temp * h_temp
                               + (idy * pixbinY + cfYDistancegpu) * w_temp
                               + (idx * pixbinX + cfXDistancegpu)]
                    + temp1 * temp2;
            }
            sumprod += prod[y * w * h + idy * w + idx];
        }

        if (isNBcalculation)
        {
            NBmeanGPU[idy * w + idx] = temp1;
            NBcovarianceGPU[idy * w + idx] =
                sumprod / prodnumarray[x * w * h + idy * w + idx]
                - temp1 * temp2;
        }

        __syncthreads();

        if (!isNBcalculation)
        {
            data1[x * w * h + idy * w + idx] = sumprod
                / (prodnumarray[x * w * h + idy * w + idx] * temp1 * temp2);
            __syncthreads();

            sumprod = 0.0;
            double sumprod2 =
                0.0; // sum of all intensity products squared; divide by num to
                     // get the average <(i(n)i(n+del))^2>
            int binct = indexarray[idy * w + idx] - ctbin;
            double tempvariable = 0.0;
            for (int y = 1; y <= binct; y++)
            {
                prodnumarray[x * w * h + idy * w + idx] = (int)floor(
                    (double)prodnumarray[x * w * h + idy * w + idx] / 2.0);

                for (int z = 0; z < prodnumarray[x * w * h + idy * w + idx];
                     z++)
                {
                    prod[z * w * h + idy * w + idx] =
                        (prod[2 * z * w * h + idy * w + idx]
                         + prod[(2 * z + 1) * w * h + idy * w + idx])
                        / 2.0;
                    __syncthreads();
                }
            }

            for (int z = 0; z < pnumgpu[idy * w + idx]; z++)
            {
                tempvariable = prod[z * w * h + idy * w + idx];
                sumprod += tempvariable; // calculate the sum of prod, i.e. the
                                         // raw correlation value ...
                sumprod2 +=
                    powf(tempvariable, 2.0); // ... and the sum of the squares
            }

            blockvararray[x * w * h + idy * w + idx] =
                (sumprod2 / pnumgpu[idy * w + idx]
                 - powf(sumprod / pnumgpu[idy * w + idx], 2.0))
                / ((pnumgpu[idy * w + idx] - 1) * powf(temp1 * temp2, 2.0));

            sdarray[x * w * h + idy * w + idx] =
                sqrt(blockvararray[x * w * h + idy * w + idx]);
        } // if (!isNBcalculation)

    } // if ((idx < w) && (idy < h))
} // calcacf2b

/* ------------------------------------------
NOTE: SEE JCudaImageJExampleKernelcalcacf3.cu
------------------------------------------ */
__global__ void calcacf3(float *data, int cfXDistancegpu, int cfYDistancegpu,
                         int blocklag, int w, int h, int w_temp, int h_temp,
                         int pixbinX, int pixbinY, int d, int correlatorp,
                         int correlatorq, int chanum, double frametimegpu,
                         double *data1, float *prod, double *prodnum,
                         double *blocksd, double *upper, double *lower,
                         int *crt, int *cr12, int *cr3, int *diffpos,
                         double *varblock0, double *varblock1,
                         double *varblock2, double *sampgpu, int *laggpu)
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y;
    int blockIndS = 0;

    __syncthreads();
    if ((idx < w) && (idy < h))
    {
        int x = 1;
        del = laggpu[x] / currentIncrement; // calculate the delay, i.e. the
                                            // correlation time
        for (int y = 0; y < numbin - del; y++)
        { // calculate the ...
            directm += data[y * w_temp * h_temp + idy * pixbinY * w_temp
                            + idx * pixbinX]; // direct And ...
            delayedm +=
                data[(y + del) * w_temp * h_temp
                     + (idy * pixbinY + cfYDistancegpu) * w_temp
                     + (idx * pixbinX + cfXDistancegpu)]; // delayed monitor
        }
        prodnum[0] = numbin - del; // number Of correlation products
        directm /=
            prodnum[0]; // calculate average Of direct And delayed monitor,
        delayedm /= prodnum[0]; // i.e. the average intesity <n(0)> And <n(tau)>

        for (int y = 0; y < prodnum[0]; y++)
        { // calculate the correlation
            prod[y * w * h + idy * w + idx] =
                data[y * w_temp * h_temp + idy * pixbinY * (w + cfXDistancegpu)
                     + idx * pixbinX]
                    * data[(y + del) * w_temp * h_temp
                           + (idy * pixbinY + cfYDistancegpu) * w_temp
                           + (idx * pixbinX + cfXDistancegpu)]
                - delayedm
                    * data[y * w_temp * h_temp + idy * pixbinY * w_temp
                           + idx * pixbinX]
                - directm
                    * data[(y + del) * w_temp * h_temp
                           + (idy * pixbinY + cfYDistancegpu) * w_temp
                           + (idx * pixbinX + cfXDistancegpu)]
                + delayedm * directm;
            __syncthreads();
            sumprod += prod[y * w * h + idy * w
                            + idx]; // calculate the sum Of prod, i.e. the raw
                                    // correlation value ...
            sumprod2 += powf(prod[y * w * h + idy * w + idx],
                             2.0); // ... And the sum Of the squares
        }

        varblock0[idy * w + idx] =
            currentIncrement * frametimegpu; // the time Of the block curve
        varblock1[idy * w + idx] =
            (sumprod2 / prodnum[0] - powf(sumprod / prodnum[0], 2.0))
            / (prodnum[0] * powf(directm * delayedm, 2.0));

        for (int y = 1; y < blocknumgpu; y++)
        { // perform blocking operations
            prodnum[y] =
                (int)floor((double)prodnum[y - 1]
                           / 2); // the number Of samples For the blocking curve
                                 // decreases by a factor 2 With every Step
            sumprod = 0;
            sumprod2 = 0;
            for (int z = 0; z < prodnum[y]; z++)
            { // bin the correlation data And calculate the blocked values for
              // the SD
                prod[z * w * h + idy * w + idx] =
                    (prod[2 * z * w * h + idy * w + idx]
                     + prod[(2 * z + 1) * w * h + idy * w + idx])
                    / 2;
                __syncthreads();
                sumprod += prod[z * w * h + idy * w + idx];
                sumprod2 += powf(prod[z * w * h + idy * w + idx], 2.0);
            }

            // This is the correct one
            varblock0[y * w * h + idy * w + idx] =
                (currentIncrement * powf(2, (double)y))
                * frametimegpu; // the time Of the block curve
            varblock1[y * w * h + idy * w + idx] =
                (sumprod2 / prodnum[y] - powf(sumprod / prodnum[y], 2.0))
                / (prodnum[y]
                   * powf(directm * delayedm, 2.0)); // value of the block curve
        }

        for (int x = 0; x < blocknumgpu; x++)
        {
            varblock1[x * w * h + idy * w + idx] =
                sqrt(varblock1[x * w * h + idy * w
                               + idx]); // calculate the standard deviation
            varblock2[x * w * h + idy * w + idx] =
                varblock1[x * w * h + idy * w + idx]
                / sqrt((double)2 * (prodnum[x] - 1)); // calculate the error
            __syncthreads();
            upper[x * w * h + idy * w + idx] =
                varblock1[x * w * h + idy * w + idx]
                + varblock2[x * w * h + idy * w
                            + idx]; // upper and lower quartile
            lower[x * w * h + idy * w + idx] =
                varblock1[x * w * h + idy * w + idx]
                - varblock2[x * w * h + idy * w + idx];
        }

        // determine index where blocking criteria are fulfilled
        for (int x = 0; x < blocknumgpu - 1; x++)
        { // do neighboring points have overlapping error bars?
            if (upper[x * w * h + idy * w + idx]
                    > lower[(x + 1) * w * h + idy * w + idx]
                && upper[(x + 1) * w * h + idy * w + idx]
                    > lower[x * w * h + idy * w + idx])
            {
                crt[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 2; x++)
        { // do three adjacent points have overlapping error bars?
            if (crt[x * w * h + idy * w + idx]
                    * crt[(x + 1) * w * h + idy * w + idx]
                == 1)
            {
                cr12[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 1; x++)
        { // do neighboring points have a positive difference (increasing SD)?
            if (varblock1[(x + 1) * w * h + idy * w + idx]
                    - varblock1[x * w * h + idy * w + idx]
                > 0)
            {
                diffpos[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 2; x++)
        { // do three neighboring points monotonically increase?
            if (diffpos[x * w * h + idy * w + idx]
                    * diffpos[(x + 1) * w * h + idy * w + idx]
                == 1)
            {
                cr3[x * w * h + idy * w + idx] = 1;
            }
        }

        for (int x = 0; x < blocknumgpu - 2; x++)
        { // find the last triple of points with monotonically increasing
          // differences and non-overlapping error bars
            if ((cr3[x * w * h + idy * w + idx] == 1
                 && cr12[x * w * h + idy * w + idx] == 0))
            {
                last0 = x;
            }
        }

        for (int x = 0; x <= last0; x++)
        { // indices of two pairs that pass criterion 1 an 2
            cr12[x * w * h + idy * w + idx] = 0;
        }

        cr12[(blocknumgpu - 3) * w * h + idy * w + idx] =
            0; // criterion 3, the last two points can't be part of the blocking
               // triple
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

// initialize/set all values in the int array to value
void initializeintarr(int *array, unsigned long size, int value)
{
    for (unsigned long i = 0; i < size; i++)
    {
        *(array + i) = value;
    }
}

// initialize/set all values in the double array to value
void initializedoublearr(double *array, unsigned long size, double value)
{
    for (unsigned long i = 0; i < size; i++)
    {
        *(array + i) = value;
    }
}

/*
 * Calls calcacf3() and calcacf2()
 *
 * Class:     gpufitImFCS_GpufitImFCS
 * Method:    calcACF
 * Signature: ([F[D[D[D[D[D[ILgpufitImFCS/GpufitImFCS/ACFParameters;)V
 */
void JNICALL Java_gpufitImFCS_GpufitImFCS_calcACF(
    JNIEnv *env, jclass cls, jfloatArray pixels, jdoubleArray pixels1,
    jdoubleArray blockvararray, jdoubleArray NBmeanGPU,
    jdoubleArray NBcovarianceGPU, jdoubleArray blocked1D,
    jdoubleArray bleachcorr_params, jdoubleArray samp, jintArray lag,
    jobject ACFInputParams)
{
    // NOTE: outputs are stored in pixels1 and blockvararray arrays.
    // NOTE: Cpixels1 and Cblockvararray are temporary arrays to store output
    // values before passing them to Java output arrays, by reference.

    // host arrays
    //------------------ common ---------------------------
    jfloat *Cpixels;
    double *Cpixels1;
    float *prod;
    double *Cbleachcorr_params;
    jdouble *Csamp;
    jint *Clag;
    double *Cblocked1D; // to copy Cpixels1 after calcacf3
    jboolean isNBcalculation;

    //------------------ calcacf3 ---------------------------
    double *prodnum;
    double *blocksd;
    double *upper;
    double *lower;
    int *crt;
    int *cr12;
    int *cr3;
    int *diffpos;
    double *varblock0;
    double *varblock1;
    double *varblock2;

    //------------------ calcacf2 ---------------------------
    int *prodnumarray;
    int *indexarray;
    double *Cblockvararray;
    double *blocksdarray;
    int *pnumgpu;

    //--------------- N & B calculations in calcacf2 --------------------
    double *CNBmeanGPU;
    double *CNBcovarianceGPU;

    // CUDA arrays
    //------------------ common ---------------------------
    float *d_Cpixels;
    double *d_Cpixels1;
    double *d_Cbleachcorr_params;
    double *d_Csamp;
    int *d_Clag;
    float *d_prod;

    //------------------ calcacf3 ---------------------------
    double *d_prodnum;
    double *d_blocksd;
    double *d_upper;
    double *d_lower;
    int *d_crt;
    int *d_cr12;
    int *d_cr3;
    int *d_diffpos;
    double *d_varblock0;
    double *d_varblock1;
    double *d_varblock2;

    //------------------ calcacf2 ---------------------------
    int *d_prodnumarray;
    int *d_indexarray;
    double *d_Cblockvararray;
    double *d_blocksdarray;
    int *d_pnumgpu;

    //--------------- N & B calculations in calcacf2 --------------------
    double *d_NBmeanGPU;
    double *d_NBcovarianceGPU;

    try
    {
        int device = 0;
        CUDA_CHECK_STATUS(cudaSetDevice(device));

        // reference:
        // https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
        //  NOTE: using stream to control synchronous processing of calcacf2 and
        //  calcacf3 kernels. based on the reference, we can potentially use it
        //  to speed up the transfer of data process.
        cudaStream_t stream;
        CUDA_CHECK_STATUS(cudaStreamCreate(&stream));

        size_t SIZEINT = sizeof(int);
        size_t SIZEFLOAT = sizeof(float);
        size_t SIZEDOUBLE = sizeof(double);

        // input arrays required for calculations.
        Cpixels = env->GetFloatArrayElements(pixels, NULL);
        Cbleachcorr_params =
            env->GetDoubleArrayElements(bleachcorr_params, NULL);
        Csamp = env->GetDoubleArrayElements(samp, NULL);
        Clag = env->GetIntArrayElements(lag, NULL);

        //      jint lensamp = env->GetArrayLength(samp);
        //      jint lenlag = env->GetArrayLength(lag);

        // get parameters that are required for the ACF calculations from the
        // ACFInputParams object
        jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

        jfieldID widthId = env->GetFieldID(ACFInputParamsCls, "width", "I");
        jfieldID heightId = env->GetFieldID(ACFInputParamsCls, "height", "I");
        jfieldID w_tempId = env->GetFieldID(ACFInputParamsCls, "w_temp", "I");
        jfieldID h_tempId = env->GetFieldID(ACFInputParamsCls, "h_temp", "I");
        jfieldID pixbinXId = env->GetFieldID(ACFInputParamsCls, "pixbinX", "I");
        jfieldID pixbinYId = env->GetFieldID(ACFInputParamsCls, "pixbinY", "I");
        jfieldID firstframeId =
            env->GetFieldID(ACFInputParamsCls, "firstframe", "I");
        jfieldID lastframeId =
            env->GetFieldID(ACFInputParamsCls, "lastframe", "I");
        jfieldID cfXDistanceId =
            env->GetFieldID(ACFInputParamsCls, "cfXDistance", "I");
        jfieldID cfYDistanceId =
            env->GetFieldID(ACFInputParamsCls, "cfYDistance", "I");
        jfieldID correlatorpId =
            env->GetFieldID(ACFInputParamsCls, "correlatorp", "D");
        jfieldID correlatorqId =
            env->GetFieldID(ACFInputParamsCls, "correlatorq", "D");
        jfieldID frametimeId =
            env->GetFieldID(ACFInputParamsCls, "frametime", "D");
        jfieldID backgroundId =
            env->GetFieldID(ACFInputParamsCls, "background", "I");
        jfieldID mtab1Id = env->GetFieldID(ACFInputParamsCls, "mtab1", "D");
        jfieldID mtabchanumminus1Id =
            env->GetFieldID(ACFInputParamsCls, "mtabchanumminus1", "D");
        jfieldID sampchanumminus1Id =
            env->GetFieldID(ACFInputParamsCls, "sampchanumminus1", "D");
        jfieldID chanumId = env->GetFieldID(ACFInputParamsCls, "chanum", "I");
        jfieldID isNBcalculationId =
            env->GetFieldID(ACFInputParamsCls, "isNBcalculation", "Z");
        jfieldID bleachcorr_gpuId =
            env->GetFieldID(ACFInputParamsCls, "bleachcorr_gpu", "Z");
        jfieldID bleachcorr_orderId =
            env->GetFieldID(ACFInputParamsCls, "bleachcorr_order", "I");

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
        jdouble correlatorpdbl =
            env->GetDoubleField(ACFInputParams, correlatorpId);
        jdouble correlatorqdbl =
            env->GetDoubleField(ACFInputParams, correlatorqId);
        jdouble frametime = env->GetDoubleField(ACFInputParams, frametimeId);
        jint background = env->GetIntField(ACFInputParams, backgroundId);
        jdouble mtab1 = env->GetDoubleField(
            ACFInputParams, mtab1Id); // mtab[1], used to calculate blocknumgpu.
        jdouble mtabchanumminus1 = env->GetDoubleField(
            ACFInputParams,
            mtabchanumminus1Id); // mtab[chanum-1], used to calculate
                                 // pnumgpu[counter_indexarray]
        jdouble sampchanumminus1 = env->GetDoubleField(
            ACFInputParams,
            sampchanumminus1Id); // samp[chanum-1], used to calculate
                                 // pnumgpu[counter_indexarray]
        jint chanum = env->GetIntField(ACFInputParams, chanumId);
        isNBcalculation =
            env->GetBooleanField(ACFInputParams, isNBcalculationId);
        jboolean bleachcorr_gpu =
            env->GetBooleanField(ACFInputParams, bleachcorr_gpuId);
        jint bleachcorr_order =
            env->GetIntField(ACFInputParams, bleachcorr_orderId);

        // initialize parameters
        int correlatorp = (int)correlatorpdbl;
        int correlatorq = (int)correlatorqdbl;
        int framediff = lastframe - firstframe + 1;
        size_t size = w_temp * h_temp * framediff * SIZEFLOAT;
        size_t size1 = width * height * chanum * SIZEDOUBLE;

        int blocklaggpu = 1;

        size_t size2 = framediff * width * height * SIZEFLOAT;
        size_t sizeblockvararray = chanum * width * height * SIZEDOUBLE;

        int blocknumgpu = (int)(floor(log(mtab1) / log(2)) - 2);

        // blockSize and gridSize
        int BLKSIZEXY = 16;
        int a = (width > height) ? width : height;
        int GRIDSIZEXY = (a + BLKSIZEXY - 1) / BLKSIZEXY;

        dim3 blockSize(BLKSIZEXY, BLKSIZEXY, 1);
        dim3 gridSize(GRIDSIZEXY, GRIDSIZEXY, 1);

        int b = (w_temp > h_temp) ? w_temp : h_temp;
        int GRIDSIZEXY_Input = (b + BLKSIZEXY - 1) / BLKSIZEXY;

        dim3 gridSize_Input(GRIDSIZEXY_Input, GRIDSIZEXY_Input, 1);

        // dynamic memory allocation and/or initialization
        //------------------ common parameters ---------------------------
        Cpixels1 = (double *)malloc(size1);
        prod = (float *)malloc(size2);
        Cblocked1D = (double *)malloc(
            size1); // Cblocked1D copies Cpixels1 array after calcacf3
                    // calculation and not required GPU memory.

        //------------------ calcacf3 ---------------------------
        // Using if (!isNBcalculation) to comment out this section will cause
        // warnings "may be used uninitialized in this function" being shown
        // during compilation instead, we will still initialize the arrays with
        // malloc, but skip the initializedoublearr, initializeintarr functions.
        prodnum = (double *)malloc(blocknumgpu * SIZEDOUBLE);
        blocksd = (double *)malloc(sizeblockvararray);
        upper = (double *)malloc(blocknumgpu * width * height * SIZEDOUBLE);
        lower = (double *)malloc(blocknumgpu * width * height * SIZEDOUBLE);
        crt = (int *)malloc((blocknumgpu - 1) * width * height * SIZEINT);
        cr12 = (int *)malloc((blocknumgpu - 2) * width * height * SIZEINT);
        cr3 = (int *)malloc((blocknumgpu - 2) * width * height * SIZEINT);
        diffpos = (int *)malloc((blocknumgpu - 1) * width * height * SIZEINT);
        varblock0 = (double *)malloc(blocknumgpu * width * height * SIZEDOUBLE);
        varblock1 = (double *)malloc(blocknumgpu * width * height * SIZEDOUBLE);
        varblock2 = (double *)malloc(blocknumgpu * width * height * SIZEDOUBLE);

        if (!isNBcalculation)
        {
            initializedoublearr(prodnum, blocknumgpu,
                                1.0); // initialize all values in prodnum to 1.0
            initializedoublearr(blocksd, chanum * width * height, 0.0);
            initializedoublearr(upper, blocknumgpu * width * height, 0.0);
            initializedoublearr(lower, blocknumgpu * width * height, 0.0);
            initializeintarr(crt, (blocknumgpu - 1) * width * height, 0);
            initializeintarr(cr12, (blocknumgpu - 2) * width * height, 0);
            initializeintarr(cr3, (blocknumgpu - 2) * width * height, 0);
            initializeintarr(diffpos, (blocknumgpu - 1) * width * height, 0);
            initializedoublearr(varblock0, blocknumgpu * width * height, 0.0);
            initializedoublearr(varblock1, blocknumgpu * width * height, 0.0);
            initializedoublearr(varblock2, blocknumgpu * width * height, 0.0);
        }

        //------------------ calcacf2 ---------------------------
        prodnumarray = (int *)malloc(chanum * width * height * SIZEINT);
        indexarray = (int *)malloc(width * height * SIZEINT);
        Cblockvararray = (double *)malloc(sizeblockvararray);
        blocksdarray = (double *)malloc(sizeblockvararray);
        pnumgpu = (int *)malloc(width * height * SIZEINT);

        //--------------- N & B calculations in calcacf2 --------------------
        CNBmeanGPU = (double *)malloc(width * height * SIZEDOUBLE);
        CNBcovarianceGPU = (double *)malloc(width * height * SIZEDOUBLE);

        if (isNBcalculation)
        {
            initializedoublearr(CNBmeanGPU, width * height, 0.0);
            initializedoublearr(CNBcovarianceGPU, width * height, 0.0);
        }

        // ------------------- perform calcacf3 calculation -------------------
        // Allocate memory on GPU for common arrays
        cudaMalloc((void **)&d_Cpixels, size);
        cudaMalloc((void **)&d_Cpixels1, size1);
        cudaMalloc((void **)&d_Cbleachcorr_params,
                   w_temp * h_temp * bleachcorr_order * SIZEDOUBLE);
        cudaMalloc((void **)&d_Csamp, chanum * SIZEDOUBLE);
        cudaMalloc((void **)&d_Clag, chanum * SIZEINT);
        cudaMalloc((void **)&d_prod, size2);

        // Allocate memory on GPU for calcacf3
        if (!isNBcalculation)
        {
            cudaMalloc((void **)&d_prodnum, blocknumgpu * SIZEDOUBLE);
            cudaMalloc((void **)&d_blocksd, sizeblockvararray);
            cudaMalloc((void **)&d_upper,
                       blocknumgpu * width * height * SIZEDOUBLE);
            cudaMalloc((void **)&d_lower,
                       blocknumgpu * width * height * SIZEDOUBLE);
            cudaMalloc((void **)&d_crt,
                       (blocknumgpu - 1) * width * height * SIZEINT);
            cudaMalloc((void **)&d_cr12,
                       (blocknumgpu - 2) * width * height * SIZEINT);
            cudaMalloc((void **)&d_cr3,
                       (blocknumgpu - 2) * width * height * SIZEINT);
            cudaMalloc((void **)&d_diffpos,
                       (blocknumgpu - 1) * width * height * SIZEINT);
            cudaMalloc((void **)&d_varblock0,
                       blocknumgpu * width * height * SIZEDOUBLE);
            cudaMalloc((void **)&d_varblock1,
                       blocknumgpu * width * height * SIZEDOUBLE);
            cudaMalloc((void **)&d_varblock2,
                       blocknumgpu * width * height * SIZEDOUBLE);
        }

        // copy to GPU for common arrays
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cpixels, Cpixels, size, cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cpixels1, Cpixels1, size1, cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cbleachcorr_params, Cbleachcorr_params,
                       w_temp * h_temp * bleachcorr_order * SIZEDOUBLE,
                       cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(cudaMemcpy(d_Csamp, Csamp, chanum * SIZEDOUBLE,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Clag, Clag, chanum * SIZEINT, cudaMemcpyHostToDevice));
        // NOTE: When pixels + prod are larger than half the memory available on
        // CPU, this cudaMemcpy function will fail. The half memory limit is
        // restriction in Java? If we comment this line, we will encounter a
        // different error when the kernel calcacf3 is run.
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_prod, prod, size2, cudaMemcpyHostToDevice));

        // copy to GPU for calcacf3
        if (!isNBcalculation)
        {
            CUDA_CHECK_STATUS(cudaMemcpy(d_prodnum, prodnum,
                                         blocknumgpu * SIZEDOUBLE,
                                         cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(d_blocksd, blocksd, sizeblockvararray,
                                         cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(
                d_upper, upper, blocknumgpu * width * height * SIZEDOUBLE,
                cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(
                d_lower, lower, blocknumgpu * width * height * SIZEDOUBLE,
                cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(
                d_crt, crt, (blocknumgpu - 1) * width * height * SIZEINT,
                cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(
                d_cr12, cr12, (blocknumgpu - 2) * width * height * SIZEINT,
                cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(
                d_cr3, cr3, (blocknumgpu - 2) * width * height * SIZEINT,
                cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(
                cudaMemcpy(d_diffpos, diffpos,
                           (blocknumgpu - 1) * width * height * SIZEINT,
                           cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(
                cudaMemcpy(d_varblock0, varblock0,
                           blocknumgpu * width * height * SIZEDOUBLE,
                           cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(
                cudaMemcpy(d_varblock1, varblock1,
                           blocknumgpu * width * height * SIZEDOUBLE,
                           cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(
                cudaMemcpy(d_varblock2, varblock2,
                           blocknumgpu * width * height * SIZEDOUBLE,
                           cudaMemcpyHostToDevice));
        }

        // running kernel for calcacf3
        if (!isNBcalculation)
        {
            if (bleachcorr_gpu)
            {
                bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(
                    d_Cpixels, w_temp, h_temp, framediff, bleachcorr_order,
                    frametime, d_Cbleachcorr_params);
                cudaDeviceSynchronize();
                CUDA_CHECK_STATUS(cudaGetLastError());
            }

            calcacf3<<<gridSize, blockSize, 0, stream>>>(
                d_Cpixels, cfXDistance, cfYDistance, blocklaggpu, width, height,
                w_temp, h_temp, pixbinX, pixbinY, framediff, correlatorp,
                correlatorq, chanum, frametime, d_Cpixels1, d_prod, d_prodnum,
                d_blocksd, d_upper, d_lower, d_crt, d_cr12, d_cr3, d_diffpos,
                d_varblock0, d_varblock1, d_varblock2, d_Csamp, d_Clag);

            cudaDeviceSynchronize();
            CUDA_CHECK_STATUS(cudaGetLastError());

            // copy memory from device to host for calcacf3
            CUDA_CHECK_STATUS(cudaMemcpy(Cpixels1, d_Cpixels1, size1,
                                         cudaMemcpyDeviceToHost));

            // CUDA release memory for calcacf3
            cudaFree(d_prodnum);
            cudaFree(d_blocksd);
            cudaFree(d_upper);
            cudaFree(d_lower);
            cudaFree(d_crt);
            cudaFree(d_cr12);
            cudaFree(d_cr3);
            cudaFree(d_diffpos);
            cudaFree(d_varblock0);
            cudaFree(d_varblock1);
            cudaFree(d_varblock2);

            // copy results in Cpixels1 to Cblocked1D. Values in Cpixels1 will
            // be overwritten in calcacf2 calculation memcpy(Cblocked1D,
            // Cpixels1, size1);
            for (int i = 0; i < width * height * chanum; i++)
            {
                *(Cblocked1D + i) = *(Cpixels1 + i);
            }

            // initialize the values in indexarray and pnumgpu
            int counter_indexarray = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    *(indexarray + counter_indexarray) =
                        (int)*(Cpixels1 + (y * width + x));

                    // minimum number of products given the used correlator
                    // structure and the blockIndex ind
                    double tempval = *(indexarray + counter_indexarray)
                        - log(sampchanumminus1) / log(2);
                    if (tempval < 0)
                    {
                        tempval = 0;
                    }
                    *(pnumgpu + counter_indexarray) =
                        (int)floor(mtabchanumminus1 / pow(2, tempval));
                    counter_indexarray = counter_indexarray + 1;
                }
            }

        } // if (!isNBcalculation)

        // ------------------- perform calcacf2 calculation -------------------

        // The pixel values in d_Cpixels has changed in calcacf3, reallocate
        // original Cpixels to d_Cpixels
        CUDA_CHECK_STATUS(
            cudaMemcpy(d_Cpixels, Cpixels, size, cudaMemcpyHostToDevice));

        // Allocate memory on GPU for calcacf2
        cudaMalloc((void **)&d_prodnumarray, chanum * width * height * SIZEINT);
        cudaMalloc((void **)&d_indexarray, width * height * SIZEINT);
        cudaMalloc((void **)&d_Cblockvararray, sizeblockvararray);
        cudaMalloc((void **)&d_blocksdarray, sizeblockvararray);
        cudaMalloc((void **)&d_pnumgpu, width * height * SIZEINT);

        // Allocate memory on GPU for N & B calculations in calcacf2
        if (isNBcalculation)
        {
            cudaMalloc((void **)&d_NBmeanGPU, width * height * SIZEDOUBLE);
            cudaMalloc((void **)&d_NBcovarianceGPU,
                       width * height * SIZEDOUBLE);
        }

        // copy to GPU for calcacf2
        CUDA_CHECK_STATUS(cudaMemcpy(d_indexarray, indexarray,
                                     width * height * SIZEINT,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(cudaMemcpy(d_prodnumarray, prodnumarray,
                                     chanum * width * height * SIZEINT,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(cudaMemcpy(d_Cblockvararray, Cblockvararray,
                                     sizeblockvararray,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(cudaMemcpy(d_blocksdarray, blocksdarray,
                                     sizeblockvararray,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_STATUS(cudaMemcpy(d_pnumgpu, pnumgpu,
                                     width * height * SIZEINT,
                                     cudaMemcpyHostToDevice));

        // copy to GPU for N & B calculatiosn in calcacf2
        if (isNBcalculation)
        {
            CUDA_CHECK_STATUS(cudaMemcpy(d_NBmeanGPU, CNBmeanGPU,
                                         width * height * SIZEDOUBLE,
                                         cudaMemcpyHostToDevice));
            CUDA_CHECK_STATUS(cudaMemcpy(d_NBcovarianceGPU, CNBcovarianceGPU,
                                         width * height * SIZEDOUBLE,
                                         cudaMemcpyHostToDevice));
        }

        // running kernel for calcacf2
        if (bleachcorr_gpu)
        {
            bleachcorrection<<<gridSize_Input, blockSize, 0, stream>>>(
                d_Cpixels, w_temp, h_temp, framediff, bleachcorr_order,
                frametime, d_Cbleachcorr_params);
            cudaDeviceSynchronize();
            CUDA_CHECK_STATUS(cudaGetLastError());
        }

        int numbin = framediff; // number Of data points When they are binned
        int currentIncrement = 1;
        int ctbin = 0;
        bool runthis;

        int calcacf2_x_start = (isNBcalculation) ? 1 : 0;
        int calcacf2_x_end = (isNBcalculation) ? 2 : chanum;

        for (int x = calcacf2_x_start; x < calcacf2_x_end; x++)
        {
            runthis = false;
            if (currentIncrement != Csamp[x])
            { // check whether the channel width has changed
                // Set the currentIncrement accordingly
                numbin = (int)floor((double)numbin / 2.0);
                currentIncrement = (int)Csamp[x];
                ctbin++;
                runthis = true;
            }

            if (runthis)
            { // check whether the channel width has changed
                calcacf2a<<<gridSize_Input, blockSize, 0, stream>>>(
                    d_Cpixels, w_temp, h_temp, numbin);
                cudaDeviceSynchronize();
                CUDA_CHECK_STATUS(cudaGetLastError());
            }

            calcacf2b<<<gridSize, blockSize, 0, stream>>>(
                d_Cpixels, cfXDistance, cfYDistance, width, height, w_temp,
                h_temp, pixbinX, pixbinY, d_Cpixels1, d_prod, d_Clag,
                d_prodnumarray, d_indexarray, d_Cblockvararray, d_blocksdarray,
                d_pnumgpu, x, numbin, currentIncrement, ctbin, isNBcalculation,
                d_NBmeanGPU, d_NBcovarianceGPU);
            cudaDeviceSynchronize();
            CUDA_CHECK_STATUS(cudaGetLastError());
        }

        // copy memory from device to host for calcacf2
        if (isNBcalculation)
        {
            CUDA_CHECK_STATUS(cudaMemcpy(CNBmeanGPU, d_NBmeanGPU,
                                         width * height * SIZEDOUBLE,
                                         cudaMemcpyDeviceToHost));
            CUDA_CHECK_STATUS(cudaMemcpy(CNBcovarianceGPU, d_NBcovarianceGPU,
                                         width * height * SIZEDOUBLE,
                                         cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDA_CHECK_STATUS(cudaMemcpy(Cpixels1, d_Cpixels1, size1,
                                         cudaMemcpyDeviceToHost));
            CUDA_CHECK_STATUS(cudaMemcpy(Cblockvararray, d_Cblockvararray,
                                         sizeblockvararray,
                                         cudaMemcpyDeviceToHost));
        }

        // CUDA release memory for calcacf2
        cudaFree(d_prodnumarray);
        cudaFree(d_indexarray);
        cudaFree(d_Cblockvararray);
        cudaFree(d_blocksdarray);
        cudaFree(d_pnumgpu);

        // CUDA release memory for N & B calculations in calcacf2
        if (isNBcalculation)
        {
            cudaFree(d_NBmeanGPU);
            cudaFree(d_NBcovarianceGPU);
        }

        // CUDA release memory for all
        cudaFree(d_Cpixels);
        cudaFree(d_Cpixels1);
        cudaFree(d_Cbleachcorr_params);
        cudaFree(d_Csamp);
        cudaFree(d_Clag);
        cudaFree(d_prod);

        CUDA_CHECK_STATUS(cudaStreamDestroy(stream));

        // Reference:
        // https://github.com/zchee/cuda-sample/blob/master/0_Simple/simpleMultiGPU/simpleMultiGPU.cu
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();

        // copy values to Java output arrays.
        env->SetDoubleArrayRegion(pixels1, 0, width * height * chanum,
                                  Cpixels1);
        env->SetDoubleArrayRegion(blockvararray, 0, chanum * width * height,
                                  Cblockvararray);
        env->SetDoubleArrayRegion(blocked1D, 0, width * height * chanum,
                                  Cblocked1D);

        if (isNBcalculation)
        {
            env->SetDoubleArrayRegion(NBmeanGPU, 0, width * height, CNBmeanGPU);
            env->SetDoubleArrayRegion(NBcovarianceGPU, 0, width * height,
                                      CNBcovarianceGPU);
        }

        // free all pointers
        free(prod);
        free(prodnum);
        free(blocksd);
        free(upper);
        free(lower);
        free(crt);
        free(cr12);
        free(cr3);
        free(diffpos);
        free(varblock0);
        free(varblock1);
        free(varblock2);
        free(indexarray);
        free(prodnumarray);
        free(Cblockvararray);
        free(blocksdarray);
        free(pnumgpu);
        free(Cpixels1);
        free(Cblocked1D);
        free(CNBmeanGPU);
        free(CNBcovarianceGPU);

        // release resources
        env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
        env->ReleaseDoubleArrayElements(bleachcorr_params, Cbleachcorr_params,
                                        0);
        env->ReleaseDoubleArrayElements(samp, Csamp, 0);
        env->ReleaseIntArrayElements(lag, Clag, 0);

        return;
    }
    catch (std::runtime_error &e)
    {
        // see: https://www.rgagnon.com/javadetails/java-0323.html
        jclass Exception = env->FindClass("java/lang/Exception");
        env->ThrowNew(Exception, e.what());

        // CUDA release memory for calcacf3
        if (!isNBcalculation)
        {
            cudaFree(d_prodnum);
            cudaFree(d_blocksd);
            cudaFree(d_upper);
            cudaFree(d_lower);
            cudaFree(d_crt);
            cudaFree(d_cr12);
            cudaFree(d_cr3);
            cudaFree(d_diffpos);
            cudaFree(d_varblock0);
            cudaFree(d_varblock1);
            cudaFree(d_varblock2);
        }

        // CUDA release memory for calcacf2
        cudaFree(d_prodnumarray);
        cudaFree(d_indexarray);
        cudaFree(d_Cblockvararray);
        cudaFree(d_blocksdarray);
        cudaFree(d_pnumgpu);

        // CUDA release memory for N & B calculations in calcacf2
        if (isNBcalculation)
        {
            cudaFree(d_NBmeanGPU);
            cudaFree(d_NBcovarianceGPU);
        }

        // CUDA release memory for all
        cudaFree(d_Cpixels);
        cudaFree(d_Cpixels1);
        cudaFree(d_Cbleachcorr_params);
        cudaFree(d_Csamp);
        cudaFree(d_Clag);
        cudaFree(d_prod);

        cudaDeviceReset();

        // free all pointers
        free(prod);
        free(prodnum);
        free(blocksd);
        free(upper);
        free(lower);
        free(crt);
        free(cr12);
        free(cr3);
        free(diffpos);
        free(varblock0);
        free(varblock1);
        free(varblock2);
        free(indexarray);
        free(prodnumarray);
        free(Cblockvararray);
        free(blocksdarray);
        free(pnumgpu);
        free(Cpixels1);
        free(Cblocked1D);
        free(CNBmeanGPU);
        free(CNBcovarianceGPU);

        // release resources
        env->ReleaseFloatArrayElements(pixels, Cpixels, 0);
        env->ReleaseDoubleArrayElements(bleachcorr_params, Cbleachcorr_params,
                                        0);
        env->ReleaseDoubleArrayElements(samp, Csamp, 0);
        env->ReleaseIntArrayElements(lag, Clag, 0);

        return;
    }
}

/* ------------------------------------------
AUTOCORRELATION SINGLE DIMENSION ARRAY CALCULATION END
------------------------------------------ */