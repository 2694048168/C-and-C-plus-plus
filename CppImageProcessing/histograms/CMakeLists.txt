# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/lena.jpg"
    "${PROJECT_SOURCE_DIR}/images/Histogram_Comparison_0.jpg"
    "${PROJECT_SOURCE_DIR}/images/Histogram_Comparison_1.jpg"
    "${PROJECT_SOURCE_DIR}/images/Histogram_Comparison_2.jpg"
    "${PROJECT_SOURCE_DIR}/images/Back_Projection_Theory0.jpg"
    "${PROJECT_SOURCE_DIR}/images/Template_Matching_Original_Image.jpg"
    "${PROJECT_SOURCE_DIR}/images/Template_Matching_Template_Image.jpg"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(HistogramEqualization)
target_sources(HistogramEqualization
    PRIVATE
        "histogram_equalization.cpp"
)
target_link_libraries(HistogramEqualization
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(HistogramCalculation)
target_sources(HistogramCalculation
    PRIVATE
        "histogram_calculation.cpp"
)
target_link_libraries(HistogramCalculation
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(HistogramComparison)
target_sources(HistogramComparison
    PRIVATE
        "histogram_comparison.cpp"
)
target_link_libraries(HistogramComparison
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(BackProjection)
target_sources(BackProjection
    PRIVATE
        "back_projection.cpp"
)
target_link_libraries(BackProjection
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(TemplateMatching)
target_sources(TemplateMatching
    PRIVATE
        "template_matching.cpp"
)
target_link_libraries(TemplateMatching
    PRIVATE
        "${OpenCV_LIBS}"
)
