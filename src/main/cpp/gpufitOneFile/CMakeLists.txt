cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(agpufit LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)


include_directories($ENV{JAVA_HOME}/include)
# Additionally, for platform-specific headers (e.g., jni_md.h)
if(APPLE)
    include_directories($ENV{JAVA_HOME}/include/darwin)
elseif(WIN32)
    include_directories($ENV{JAVA_HOME}/include/win32)
else()
    include_directories($ENV{JAVA_HOME}/include/linux)
endif()

# Define the library this folder will build
add_library(agpufit SHARED
    agpufitjni.cu
    )