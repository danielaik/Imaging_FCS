#include <jni.h>

class ACFInputParamsWrapper {
public:
    jint width;
    jint height;
    jint w_temp;
    jint h_temp;
    jint pixbinX;
    jint pixbinY;
    jint firstframe;
    jint lastframe;
    jint cfXDistance;
    jint cfYDistance;
    jdouble correlatorp;
    jdouble correlatorq;
    jdouble frametime;
    jint background;
    jdouble mtab1;
    jdouble mtabchanumminus1;
    jdouble sampchanumminus1;
    jint chanum;
    jboolean isNBcalculation;
    jboolean bleachcorr_gpu;
    jint bleachcorr_order;

    ACFInputParamsWrapper(JNIEnv *env, jobject ACFInputParams);
};