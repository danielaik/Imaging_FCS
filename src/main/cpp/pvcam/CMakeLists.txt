cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(pvcam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
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

link_directories(dependencies/lib/amd64)

# Define the library this folder will build
add_library(JNIPhotometricsPVCAMsdk SHARED
    dependencies/include/directCameraReadout_pvcamsdk_Photometrics_PVCAM_SDK.h
    dependencies/include/master.h
    dependencies/include/pvcam.h
    Common.cpp
    Common.h
    ImageClsPVCam.cpp
    ImageClsPVCam.h
    Source.cpp
    TimeKeeper.cpp
    TimeKeeper.h
    version.h
    )

target_link_libraries(JNIPhotometricsPVCAMsdk pvcam64)