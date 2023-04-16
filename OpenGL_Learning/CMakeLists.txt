cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(LearningOpenGL
    VERSION 0.1.1
    DESCRIPTION "The Tutorials of Learning OpenGL"
    LANGUAGES CXX C
)
# glad.c 一定需要 C 编译器,否则链接阶段无法找到 '_glad_' 一系列定义

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()

# ==========================================================
add_subdirectory("${PROJECT_SOURCE_DIR}/00_hello_opengl/")
add_subdirectory("${PROJECT_SOURCE_DIR}/01_hello_triangle/")
add_subdirectory("${PROJECT_SOURCE_DIR}/02_rectangular/")
add_subdirectory("${PROJECT_SOURCE_DIR}/03_shader/")
add_subdirectory("${PROJECT_SOURCE_DIR}/04_uniform/")
add_subdirectory("${PROJECT_SOURCE_DIR}/05_attribute/")
add_subdirectory("${PROJECT_SOURCE_DIR}/06_texture/")
add_subdirectory("${PROJECT_SOURCE_DIR}/07_transformations/")
add_subdirectory("${PROJECT_SOURCE_DIR}/08_coordinate/")
add_subdirectory("${PROJECT_SOURCE_DIR}/09_camera/")
