include_directories(dependencies/include)
link_directories(dependencies/lib)

# Define the library this folder will build
add_library(JNIAndorSDK2v3 SHARED
    dependencies/include/ATMCD32D.H
    dependencies/include/fiji_plugin_imaging_fcs_directCameraReadout_andorsdk2v3_AndorSDK2v3.h
    ImageClsSDK2.cpp
    ImageClsSDK2.h
    sourcesdk2.cpp
    ${TIMEKEEPER}
    )

target_link_libraries(JNIAndorSDK2v3 atmcd64m)