cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cameras LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include java directories
include_directories($ENV{JAVA_HOME}/include)
# Additionally, for platform-specific headers (e.g., jni_md.h)
if(APPLE)
    include_directories($ENV{JAVA_HOME}/include/darwin)
elseif(WIN32)
    include_directories($ENV{JAVA_HOME}/include/win32)
else()
    include_directories($ENV{JAVA_HOME}/include/linux)
endif()

set(TIMEKEEPER ${CMAKE_CURRENT_SOURCE_DIR}/TimeKeeper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/TimeKeeper.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add modules as libraries
add_subdirectory(dcam)
add_subdirectory(pvcam)
add_subdirectory(sdk2)
add_subdirectory(sdk3)