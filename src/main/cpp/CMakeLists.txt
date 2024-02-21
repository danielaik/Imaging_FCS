cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Imaging_FCS)

# Include directories
if (WIN32)
    add_subdirectory(dcam)
    add_subdirectory(pvcam)
    add_subdirectory(sdk2)
    add_subdirectory(sdk3)
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    add_subdirectory(gpufitOneFile)
endif()