#include "acf_input_params_wrapper.h"

ACFInputParamsWrapper::ACFInputParamsWrapper(JNIEnv *env, jobject ACFInputParams)
{
    jclass ACFInputParamsCls = env->GetObjectClass(ACFInputParams);

    // Retrieve field IDs and values
    width = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "width", "I"));
    height = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "height", "I"));
    w_temp = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "w_temp", "I"));
    h_temp = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "h_temp", "I"));
    pixbinX = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "pixbinX", "I"));
    pixbinY = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "pixbinY", "I"));
    firstframe = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "firstframe", "I"));
    lastframe = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "lastframe", "I"));
    cfXDistance = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "cfXDistance", "I"));
    cfYDistance = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "cfYDistance", "I"));
    correlatorp = env->GetDoubleField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "correlatorp", "D"));
    correlatorq = env->GetDoubleField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "correlatorq", "D"));
    frametime = env->GetDoubleField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "frametime", "D"));
    background = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "background", "I"));
    mtab1 = env->GetDoubleField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "mtab1", "D"));
    mtabchanumminus1 = env->GetDoubleField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "mtabchanumminus1", "D"));
    sampchanumminus1 = env->GetDoubleField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "sampchanumminus1", "D"));
    chanum = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "chanum", "I"));
    isNBcalculation = env->GetBooleanField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "isNBcalculation", "Z"));
    bleachcorr_gpu = env->GetBooleanField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "bleachcorr_gpu", "Z"));
    bleachcorr_order = env->GetIntField(ACFInputParams, env->GetFieldID(ACFInputParamsCls, "bleachcorr_order", "I"));
}