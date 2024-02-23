include_directories(dependencies/include)
link_directories(dependencies/lib)

# Define the library this folder will build
add_library(JNIPhotometricsPVCAMsdk SHARED
    dependencies/include/fiji_plugin_imaging_fcs_directCameraReadout_pvcamsdk_Photometrics_PVCAM_SDK.h
    dependencies/include/master.h
    dependencies/include/pvcam.h
    Common.cpp
    Common.h
    ImageClsPVCam.cpp
    ImageClsPVCam.h
    Source.cpp
    version.h
    ${TIMEKEEPER}
    )

target_link_libraries(JNIPhotometricsPVCAMsdk pvcam64)