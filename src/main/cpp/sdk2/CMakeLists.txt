cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sdk2 LANGUAGES CXX)

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

link_directories(dependencies/lib)

# Define the library this folder will build
add_library(JNIAndorSDK2v3 SHARED
    dependencies/include/ATMCD32D.H
    dependencies/include/directCameraReadout_andorsdk2v3_AndorSDK2v3.h
    ImageClsSDK2.cpp
    ImageClsSDK2.h
    sourcesdk2.cpp
    TimeKeeper.cpp
    TimeKeeper.h
    )

target_link_libraries(JNIAndorSDK2v3 atmcd64m)