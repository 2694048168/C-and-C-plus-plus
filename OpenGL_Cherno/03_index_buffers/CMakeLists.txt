cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()

# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/libs)

include_directories(${PROJECT_SOURCE_DIR}/external/GLFW/include)
link_directories(${PROJECT_SOURCE_DIR}/external/GLFW/lib-vc2022)

# the GLEW binary download only for VC++, not for MinGW
include_directories(${PROJECT_SOURCE_DIR}/external/GLEW/include)
link_directories(${PROJECT_SOURCE_DIR}/external/GLEW/lib/Release/x64)
# to ensure static library(glew32s.lib) building with GLEW
add_compile_definitions(GLEW_STATIC)

add_executable(square src/square.cpp)
target_link_libraries(square
    glfw3.lib
    opengl32.lib
    glew32s.lib
)
