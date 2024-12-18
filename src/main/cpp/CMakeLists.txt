cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Imaging_FCS)

# Include directories
if(WIN32)
    add_subdirectory(cameras)
endif()

# Find CUDA
find_package(CUDA)

if(CUDA_FOUND)
    add_subdirectory(gpufit)
endif()