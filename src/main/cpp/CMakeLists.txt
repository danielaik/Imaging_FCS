cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Imaging_FCS)

# Include directories
if (WIN32)
    add_subdirectory(cameras)
endif()

# Find CUDA
find_package(CUDAToolkit)

# Add subdirectory gpufit (requires CUDA)
if (CUDAToolkit_FOUND)
    add_subdirectory(gpufit)
endif()