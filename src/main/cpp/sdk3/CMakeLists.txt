cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sdk3 LANGUAGES CXX)

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
add_library(JNIAndorSDK3v2 SHARED
    dependencies/include/atcore.h
    dependencies/include/ATMCD32D.H
    dependencies/include/atutility.h
    dependencies/include/directCameraReadout_andorsdk3v2_AndorSDK3v2.h
    ImageClsSDK3.cpp
    ImageClsSDK3.h
    sourcesdk3v2.cpp
    TimeKeeper.cpp
    TimeKeeper.h
    )

target_link_libraries(JNIAndorSDK3v2 atcorem atmcd64m atutilitym)