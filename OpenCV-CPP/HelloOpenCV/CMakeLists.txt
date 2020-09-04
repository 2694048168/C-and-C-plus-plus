# 要求 CMake 最低版本
cmake_minimum_required (VERSION 3.10)

# 项目名称
project (HelloOpenCV)

# 设置编译选项
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++17")
# 设置编译模式, Debug、Release
set (CMake_BUILD_TYPE "Debug")

# 设置源文件路径以及编译的源文件
set (source_dir "${PROJECT_SOURCE_DIR}/src/")
file (GLOB source_files "${source_dir}/HelloOpenCV.cpp")

# 设置输出二进制路径
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 第三方库的使用 OpenCV
# 设置 OpenCV 库的路径，能有搜索到类似 OpenCVConfig.cmake 配置文件
# OpenCVConfig.cmake 配置文件是编译第三方库的时候自动生成的
set (OpenCV_DIR "./../OpenCV-MinGW-Build-OpenCV-4.1.1-x64/")
# 自动查找
find_package(OpenCV REQUIRED)
# 使用 OpenCVConfig.cmake 配置里面的变量添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 编译源文件生成可执行程序
add_executable (HelloOpenCV ${source_files})

# 使用 OpenCVConfig.cmake 配置里面的变量链接库文件
target_link_libraries(HelloOpenCV ${OpenCV_LIBS})
