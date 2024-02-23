include_directories(dependencies/include)
link_directories(dependencies/lib/win64)

# Define the library this folder will build
add_library(dcam SHARED
    dependencies/include/dcamapi4.h
    dependencies/include/dcamprop.h
    dependencies/include/fiji_plugin_imaging_fcs_directCameraReadout_hamadcamsdk4_Hamamatsu_DCAM_SDK4.h
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
    ${TIMEKEEPER}
    )

target_link_libraries(dcam dcamapi)