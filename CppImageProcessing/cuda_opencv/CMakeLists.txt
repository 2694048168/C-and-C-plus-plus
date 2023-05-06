# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/cat.jpg"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(CudaOpenCV)
target_sources(CudaOpenCV
    PRIVATE
        "metrics.cpp"
)
target_link_libraries(CudaOpenCV
    PRIVATE
        "${OpenCV_LIBS}"
)
