# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/cards.png"
    "${PROJECT_SOURCE_DIR}/images/original.jpg"
    "${PROJECT_SOURCE_DIR}/images/motion_original.jpg"
    "${PROJECT_SOURCE_DIR}/images/gst_input.jpg"
    "${PROJECT_SOURCE_DIR}/images/period_input.jpg"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(ImageSegmentation)
target_sources(ImageSegmentation
    PRIVATE
        "segmentation.cpp"
)
target_link_libraries(ImageSegmentation
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(OutFocusDeblurFilter)
target_sources(OutFocusDeblurFilter
    PRIVATE
        "focus_deblur.cpp"
)
target_link_libraries(OutFocusDeblurFilter
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(MotionDeblurFilter)
target_sources(MotionDeblurFilter
    PRIVATE
        "motion_deblur.cpp"
)
target_link_libraries(MotionDeblurFilter
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(AnisotropicGradient)
target_sources(AnisotropicGradient
    PRIVATE
        "anisotropic.cpp"
)
target_link_libraries(AnisotropicGradient
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(PeriodicNoise)
target_sources(PeriodicNoise
    PRIVATE
        "periodic_noise.cpp"
)
target_link_libraries(PeriodicNoise
    PRIVATE
        "${OpenCV_LIBS}"
)
