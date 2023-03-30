cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(CUDA_PROGRAMMING
    VERSION 0.1.1
    DESCRIPTION "The CUDA Programming with C++"
    LANGUAGES CXX CUDA
)

# CUDA 12 Features: Support Visual Studio 2022
# https://www.bookstack.cn/read/Modern-CMake-CN-Modern-CMake-zh_CN/
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
message(STATUS "There is CUDA toolkits: ${CMAKE_CUDA_COMPILER}")
message(STATUS "The CUDA toolkits is: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "The CUDA Version is: ${CMAKE_CUDA_COMPILER_VERSION}")

# ===============================================================
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/01_kernelFunction/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/02_multipleThread/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/03_multipleDimGrid/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/04_CudaPipeline/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/05_CudaError/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/06_CudaEventPerformance/)
# nvprof cudaPerformance.exe
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/07_cudaThrust/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/08_cuBLAS/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/09_cuSolver/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/10_cuRAND/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/11_cuFFT/)
