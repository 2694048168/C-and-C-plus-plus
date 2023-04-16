set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 把需要拷贝的资源路径都放在 "ASSETS" 变量里面
file(GLOB ASSETS 
    "${PROJECT_SOURCE_DIR}/resources/textures/container.jpg"
    "${PROJECT_SOURCE_DIR}/resources/textures/awesomeface.png"
)
# 把 "ASSETS" 指代的所有内容都拷贝到可执行文件目录下
file(COPY ${ASSETS} DESTINATION ${EXECUTABLE_OUTPUT_PATH})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")

# =========== thirdparty library ===========
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/GLAD/include/")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/GLFW/include/")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/GLM/")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/STB/")

# cmake . -B build -G "MinGW Makefiles"
# cmake . -B build -G "Visual Studio 17 2022" -DIS_MinGW=OFF
# cmake --build build
option(IS_MinGW "Is using the MinGW Compiler on Windows" ON)
if(IS_MinGW)
    link_directories("${PROJECT_SOURCE_DIR}/thirdparty/GLFW/lib-mingw-w64/")
else()
    link_directories("${PROJECT_SOURCE_DIR}/thirdparty/GLFW/lib-vc2022/")
endif()

add_executable(coordinate)
target_sources(coordinate
    PRIVATE
        "src/coordinate.cpp"
        "src/shader.cpp"
        "src/utils.cpp"
        "${PROJECT_SOURCE_DIR}/thirdparty/GLAD/src/glad.c"
)
target_link_libraries(coordinate
    glfw3.lib
    OpenGL32.Lib 
)

add_executable(coordinate_depth)
target_sources(coordinate_depth
    PRIVATE
        "src/coordinate_depth.cpp"
        "src/shader.cpp"
        "src/utils.cpp"
        "${PROJECT_SOURCE_DIR}/thirdparty/GLAD/src/glad.c"
)
target_link_libraries(coordinate_depth
    glfw3.lib
    OpenGL32.Lib 
)

add_executable(coordinate_depth_multiple)
target_sources(coordinate_depth_multiple
    PRIVATE
        "src/coordinate_depth_multiple.cpp"
        "src/shader.cpp"
        "src/utils.cpp"
        "${PROJECT_SOURCE_DIR}/thirdparty/GLAD/src/glad.c"
)
target_link_libraries(coordinate_depth_multiple
    glfw3.lib
    OpenGL32.Lib 
)
