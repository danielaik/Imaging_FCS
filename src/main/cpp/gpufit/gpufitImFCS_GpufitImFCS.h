#ifndef GPUFIT_IMFCS_H_INCLUDED
#define GPUFIT_IMFCS_H_INCLUDED

#include <jni.h>

#ifdef __cplusplus
extern "C"
{
#endif
    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    fit
     * Signature:
     * (IILjava/nio/FloatBuffer;Ljava/nio/FloatBuffer;ILjava/nio/FloatBuffer;FIILjava/nio/IntBuffer;IILjava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)I
     */
    JNIEXPORT jint JNICALL Java_gpufitImFCS_GpufitImFCS_fit(
        JNIEnv *, jclass, jint, jint, jobject, jobject, jint, jobject, jfloat,
        jint, jint, jobject, jint, jint, jobject, jobject, jobject, jobject,
        jobject);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    getLastError
     * Signature: ()Ljava/lang/String;
     */
    JNIEXPORT jstring JNICALL
    Java_gpufitImFCS_GpufitImFCS_getLastError(JNIEnv *, jclass);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    isCudaAvailableInt
     * Signature: ()Z
     */
    JNIEXPORT jboolean JNICALL
    Java_gpufitImFCS_GpufitImFCS_isCudaAvailableInt(JNIEnv *, jclass);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    getCudaVersionAsArray
     * Signature: ()[I
     */
    JNIEXPORT jintArray JNICALL
    Java_gpufitImFCS_GpufitImFCS_getCudaVersionAsArray(JNIEnv *, jclass);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    resetGPU
     * Signature: ()V
     */
    JNIEXPORT void JNICALL Java_gpufitImFCS_GpufitImFCS_resetGPU(JNIEnv *,
                                                                 jclass);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    calcDataBleachCorrection
     * Signature: ([F[FLgpufitImFCS/GpufitImFCS/ACFParameters;)V
     */
    JNIEXPORT void JNICALL
    Java_gpufitImFCS_GpufitImFCS_calcDataBleachCorrection(JNIEnv *, jclass,
                                                          jfloatArray,
                                                          jfloatArray, jobject);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    isBinningMemorySufficient
     * Signature: (LgpufitImFCS/GpufitImFCS/ACFParameters;)Z
     */
    JNIEXPORT jboolean JNICALL
    Java_gpufitImFCS_GpufitImFCS_isBinningMemorySufficient(JNIEnv *, jclass,
                                                           jobject);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    calcBinning
     * Signature: ([F[FLgpufitImFCS/GpufitImFCS/ACFParameters;)V
     */
    JNIEXPORT void JNICALL Java_gpufitImFCS_GpufitImFCS_calcBinning(
        JNIEnv *, jclass, jfloatArray, jfloatArray, jobject);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    isACFmemorySufficient
     * Signature: (LgpufitImFCS/GpufitImFCS/ACFParameters;)Z
     */
    JNIEXPORT jboolean JNICALL
    Java_gpufitImFCS_GpufitImFCS_isACFmemorySufficient(JNIEnv *, jclass,
                                                       jobject);

    /*
     * Class:     gpufitImFCS_GpufitImFCS
     * Method:    calcACF
     * Signature: ([F[D[D[D[D[D[D[D[ILgpufitImFCS/GpufitImFCS/ACFParameters;)V
     */
    JNIEXPORT void JNICALL Java_gpufitImFCS_GpufitImFCS_calcACF(
        JNIEnv *, jclass, jfloatArray, jdoubleArray, jdoubleArray, jdoubleArray,
        jdoubleArray, jdoubleArray, jdoubleArray, jdoubleArray, jintArray,
        jobject);

#ifdef __cplusplus
}
#endif

#endif