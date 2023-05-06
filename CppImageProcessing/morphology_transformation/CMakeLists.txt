# CMake and CMakeTool to Debugging with comline args
# add following configuration in '.vscode/settings.json'
# "cmake.debugConfig": {
#     "args": [
#         "../dog.jpg",
#     ]
# },

# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/LinuxLogo.jpg"
    "${PROJECT_SOURCE_DIR}/images/baboon.jpg"
    "${PROJECT_SOURCE_DIR}/images/notes.png"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(ErosionDilation)
target_sources(ErosionDilation
    PRIVATE
        "erosion_dilation.cpp"
)
target_link_libraries(ErosionDilation
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Morphology)
target_sources(Morphology
    PRIVATE
        "morphology.cpp"
)
target_link_libraries(Morphology
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(HitMiss)
target_sources(HitMiss
    PRIVATE
        "HitMiss.cpp"
)
target_link_libraries(HitMiss
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(Extract)
target_sources(Extract
    PRIVATE
        "extract_pattern.cpp"
)
target_link_libraries(Extract
    PRIVATE
        "${OpenCV_LIBS}"
)
