cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------
project(CPP_Tutorial
    VERSION 0.1.1
    DESCRIPTION "the CPP tutorial with CMake"
    LANGUAGES CXX C
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# ----------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week01/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week02/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week03/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week04/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week05/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week06/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week07/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week09/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week10/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week11/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week12/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week13/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week14/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/week15/)
