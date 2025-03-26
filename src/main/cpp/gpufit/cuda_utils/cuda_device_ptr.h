#ifndef CUDA_UTILS_CUDA_DEVICE_PTR_H
#define CUDA_UTILS_CUDA_DEVICE_PTR_H

#include <cuda_runtime.h>
#include <stdexcept>

namespace cuda_utils
{
    template <typename T>
    class CudaDevicePtr
    {
    public:
        // Default constructor
        CudaDevicePtr()
            : ptr_(nullptr)
        {}

        explicit CudaDevicePtr(size_t size)
        {
            if (cudaMalloc(&ptr_, size * sizeof(T)) != cudaSuccess)
            {
                throw std::runtime_error("Failed to allocate device memory");
            }
        }

        ~CudaDevicePtr()
        {
            if (ptr_)
            {
                cudaFree(ptr_);
            }
        }

        T *get() const
        {
            return ptr_;
        }

        // Disable copy semantics
        CudaDevicePtr(const CudaDevicePtr &) = delete;
        CudaDevicePtr &operator=(const CudaDevicePtr &) = delete;

        // Enable move semantics
        CudaDevicePtr(CudaDevicePtr &&other) noexcept
            : ptr_(other.ptr_)
        {
            other.ptr_ = nullptr;
        }

        CudaDevicePtr &operator=(CudaDevicePtr &&other) noexcept
        {
            if (this != &other)
            {
                if (ptr_)
                {
                    cudaFree(ptr_);
                }
                ptr_ = other.ptr_;
                other.ptr_ = nullptr;
            }
            return *this;
        }

    private:
        T *ptr_ = nullptr;
    };

} // namespace cuda_utils

#endif // CUDA_UTILS_CUDA_DEVICE_PTR_H
