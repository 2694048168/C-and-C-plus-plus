# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/cat.jpg"
    "${PROJECT_SOURCE_DIR}/images/lena.jpg"
    "${PROJECT_SOURCE_DIR}/images/fruits.jpg"
    "${PROJECT_SOURCE_DIR}/images/sudoku.png"
    "${PROJECT_SOURCE_DIR}/images/smarties.png"
    "${PROJECT_SOURCE_DIR}/images/generalized_hough_mini_image.jpg"
    "${PROJECT_SOURCE_DIR}/images/generalized_hough_mini_template.jpg"
    "${PROJECT_SOURCE_DIR}/images/chicky_512.png"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(Filter)
target_sources(Filter
    PRIVATE
        "linear_filter.cpp"
)
target_link_libraries(Filter
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Sobel)
target_sources(Sobel
    PRIVATE
        "sobel.cpp"
)
target_link_libraries(Sobel
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Laplacian)
target_sources(Laplacian
    PRIVATE
        "laplacian.cpp"
)
target_link_libraries(Laplacian
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Canny)
target_sources(Canny
    PRIVATE
        "canny.cpp"
)
target_link_libraries(Canny
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(HoughLine)
target_sources(HoughLine
    PRIVATE
        "hough_line.cpp"
)
target_link_libraries(HoughLine
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(HoughCircle)
target_sources(HoughCircle
    PRIVATE
        "hough_circle.cpp"
)
target_link_libraries(HoughCircle
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(TemplateDetect)
target_sources(TemplateDetect
    PRIVATE
        "template_detect.cpp"
)
target_link_libraries(TemplateDetect
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Remapping)
target_sources(Remapping
    PRIVATE
        "remapping.cpp"
)
target_link_libraries(Remapping
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Affine)
target_sources(Affine
    PRIVATE
        "affine.cpp"
)
target_link_libraries(Affine
    PRIVATE
        "${OpenCV_LIBS}"
)
