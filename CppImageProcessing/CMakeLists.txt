cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# vcpkg configuration 需要根据平台进行配置
# "vcpkg help triplet" command: Available architecture triplets
set(CMAKE_TOOLCHAIN_FILE "D:/Development/vcpkg/scripts/buildsystems/vcpkg.cmake")
# https://learn.microsoft.com/en-us/vcpkg/reference/vcpkg-json
# the configuration JSON file 'vcpkg.json'
# ==============================================================

project(CppImageProcessing
    VERSION 0.1.1
    DESCRIPTION "the Image Processing vith modern C++"
    LANGUAGES CXX
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
set(CMAKE_BUILD_TYPE "Release") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()

# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# =========== thirdparty library ===========
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ----------------------------------------------------------
# cmake -B build
# cmake --build build
# ----------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hello_start/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/geometric_drawing/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/morphology_transformation/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/image_pyramids/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/transformations/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/histograms/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/contours/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/image_segmentation/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda_opencv/)
