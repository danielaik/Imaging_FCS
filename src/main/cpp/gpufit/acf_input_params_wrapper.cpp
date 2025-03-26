#include "acf_input_params_wrapper.h"

ACFInputParamsWrapper::ACFInputParamsWrapper(JNIEnv *env, jobject ACFInputParams)
{
    jclass cls = env->GetObjectClass(ACFInputParams);

    // Retrieve field IDs and values
    width = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "width", "I"));
    height = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "height", "I"));
    w_temp = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "w_temp", "I"));
    h_temp = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "h_temp", "I"));
    pixbinX = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "pixbinX", "I"));
    pixbinY = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "pixbinY", "I"));
    firstframe = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "firstframe", "I"));
    lastframe = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "lastframe", "I"));
    cfXDistance = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "cfXDistance", "I"));
    cfYDistance = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "cfYDistance", "I"));
    correlatorp = env->GetDoubleField(ACFInputParams, env->GetFieldID(cls, "correlatorp", "D"));
    correlatorq = env->GetDoubleField(ACFInputParams, env->GetFieldID(cls, "correlatorq", "D"));
    frametime = env->GetDoubleField(ACFInputParams, env->GetFieldID(cls, "frametime", "D"));
    background = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "background", "I"));
    mtab1 = env->GetDoubleField(ACFInputParams, env->GetFieldID(cls, "mtab1", "D"));
    mtabchanumminus1 = env->GetDoubleField(ACFInputParams, env->GetFieldID(cls, "mtabchanumminus1", "D"));
    sampchanumminus1 = env->GetDoubleField(ACFInputParams, env->GetFieldID(cls, "sampchanumminus1", "D"));
    chanum = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "chanum", "I"));
    isNBcalculation = env->GetBooleanField(ACFInputParams, env->GetFieldID(cls, "isNBcalculation", "Z"));
    bleachcorr_gpu = env->GetBooleanField(ACFInputParams, env->GetFieldID(cls, "bleachcorr_gpu", "Z"));
    bleachcorr_order = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "bleachcorr_order", "I"));
    nopit = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "nopit", "I"));
    ave = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "ave", "I"));
    win_star = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "win_star", "I"));
    hin_star = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "hin_star", "I"));
    binningX = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "binningX", "I"));
    binningY = env->GetIntField(ACFInputParams, env->GetFieldID(cls, "binningY", "I"));
}