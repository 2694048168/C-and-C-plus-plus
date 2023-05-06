# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/lena.jpg"
    "${PROJECT_SOURCE_DIR}/images/cat.jpg"
    "${PROJECT_SOURCE_DIR}/images/dog.jpg"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(GeometricDrawing)
target_sources(GeometricDrawing
    PRIVATE
        "geometric.cpp"
)
target_link_libraries(GeometricDrawing
    PRIVATE
        ${OpenCV_LIBS}
)

add_executable(RandomText)
target_sources(RandomText
    PRIVATE
        "generator.cpp"
)
target_link_libraries(RandomText
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(ImageFilter)
target_sources(ImageFilter
    PRIVATE
        "filters.cpp"
)
target_link_libraries(ImageFilter
    PRIVATE
        "${OpenCV_LIBS}"
)
