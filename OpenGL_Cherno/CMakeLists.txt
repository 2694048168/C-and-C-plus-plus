cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(OpenGL_Cherno
    VERSION 0.1.1
    DESCRIPTION "The OpenGL by Cherno"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()

# ========================================================
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/00_opengl/")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/01_triangle/")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/02_shader/")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/03_index_buffers/")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/04_error/")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/05_uniforms/")
