# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/chicky_512.png"
    "${PROJECT_SOURCE_DIR}/images/stuff.jpg"
    "${PROJECT_SOURCE_DIR}/images/young_man_company_work.mp4"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(Pyramids)
target_sources(Pyramids
    PRIVATE
        "pyramids.cpp"
)
target_link_libraries(Pyramids
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Threshold)
target_sources(Threshold
    PRIVATE
        "threshold.cpp"
)
target_link_libraries(Threshold
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(inRange)
target_sources(inRange
    PRIVATE
        "inRange.cpp"
)
target_link_libraries(inRange
    PRIVATE
        "${OpenCV_LIBS}"
)
