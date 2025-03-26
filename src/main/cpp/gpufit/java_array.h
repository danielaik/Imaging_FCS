#ifndef JAVA_ARRAY_H
#define JAVA_ARRAY_H

#include <jni.h>
#include <stdexcept>
#include <type_traits>

// Base template (not defined)
template <typename ArrayType, typename ElementType>
class JavaArray;

// Specialization for jfloatArray and jfloat
template <>
class JavaArray<jfloatArray, jfloat> {
public:
    JavaArray(JNIEnv *env, jfloatArray array)
        : env_(env), array_(array), elements_(nullptr) {
        elements_ = env_->GetFloatArrayElements(array_, nullptr);
        if (elements_ == nullptr) {
            throw std::runtime_error("Failed to get jfloatArray elements");
        }
    }

    ~JavaArray() {
        env_->ReleaseFloatArrayElements(array_, elements_, 0);
    }

    jfloat* elements() const {
        return elements_;
    }

private:
    JNIEnv *env_;
    jfloatArray array_;
    jfloat *elements_;
};

// Specialization for jdoubleArray and jdouble
template <>
class JavaArray<jdoubleArray, jdouble> {
public:
    JavaArray(JNIEnv *env, jdoubleArray array)
        : env_(env), array_(array), elements_(nullptr) {
        elements_ = env_->GetDoubleArrayElements(array_, nullptr);
        if (elements_ == nullptr) {
            throw std::runtime_error("Failed to get jdoubleArray elements");
        }
    }

    ~JavaArray() {
        env_->ReleaseDoubleArrayElements(array_, elements_, 0);
    }

    jdouble* elements() const {
        return elements_;
    }

private:
    JNIEnv *env_;
    jdoubleArray array_;
    jdouble *elements_;
};

// Specialization for jintArray and jint
template <>
class JavaArray<jintArray, jint> {
public:
    JavaArray(JNIEnv *env, jintArray array)
        : env_(env), array_(array), elements_(nullptr) {
        elements_ = env_->GetIntArrayElements(array_, nullptr);
        if (elements_ == nullptr) {
            throw std::runtime_error("Failed to get jintArray elements");
        }
    }

    ~JavaArray() {
        env_->ReleaseIntArrayElements(array_, elements_, 0);
    }

    jint* elements() const {
        return elements_;
    }

private:
    JNIEnv *env_;
    jintArray array_;
    jint *elements_;
};

#endif // JAVA_ARRAY_H