# CMake needs
cmake_minimum_required(VERSION 3.10)

# Define project name
project(addingImage)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++17")
# 设置编译模式, Debug、Release
set(CMake_BUILD_TYPE "Debug")

# 设置源文件路径以及编译的源文件
set(source_dir "${PROJECT_SOURCE_DIR}/src/")
file(GLOB source_files "${source_dir}/*.cpp")
message(STATUS "Project: ${addingImage}")
message(STATUS "File   : ${source_dir}")

# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set(OpenCV_DIR "./../OpenCV-MinGW-Build-OpenCV-4.1.1-x64/")
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

# Declare the executable target built from your sources
add_executable(addingImage ${source_files})

# Link your application with OpenCV libraries
target_link_libraries(addingImage ${OpenCV_LIBS})
