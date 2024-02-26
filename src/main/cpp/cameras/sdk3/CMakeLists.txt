include_directories(dependencies/include)
link_directories(dependencies/lib)

# Define the library this folder will build
add_library(JNIAndorSDK3v2 SHARED
    dependencies/include/atcore.h
    dependencies/include/ATMCD32D.H
    dependencies/include/atutility.h
    dependencies/include/fiji_plugin_imaging_fcs_directCameraReadout_andorsdk3v2_AndorSDK3v2.h
    ImageClsSDK3.cpp
    ImageClsSDK3.h
    sourcesdk3v2.cpp
    ${TIMEKEEPER}
    )

target_link_libraries(JNIAndorSDK3v2 atcorem atmcd64m atutilitym)