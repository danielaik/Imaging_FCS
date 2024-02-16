cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(dcam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(dependencies/include)
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
add_library(dcam SHARED
    dependencies/include/dcamapi4.h
    dependencies/include/dcamprop.h
    dependencies/include/directCameraReadout_hamadcamsdk4_Hamamatsu_DCAM_SDK4.h
    common.cpp
    common.h
    console4.h
    ImageClsSDK4.cpp
    ImageClsSDK4.h
    OrcaJob.cpp
    Orcajob.h
    Source.cpp
    Stoppable.cpp
    Stoppable.h
    TimeKeeper.cpp
    TimeKeeper.h
    )