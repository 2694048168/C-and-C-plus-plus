# step1 CMakeLists.txt
# 1、工程配置部分
# 指定CMake最低版本号要求
cmake_minimum_required (VERSION 3.5)
# 项目信息
project (the_contact_managerment_system)
# 设置编译版本 如“-std=c++11”、“-std=c++14”、“-std=c++17”、“-std=c++20”
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++11")
# 设置编译模式,如“Debug”、“Release”
set (CMake_BUILD_TYPE "Debug")

# 2、依赖执行部分
# 设置源代码路径
set (source_dir "${PROJECT_SOURCE_DIR}/src/")
# 设置需要编译的源文件
file (GLOB source_files "${source_dir}/*.cpp")
# 设置编译的输出目标文件
add_executable (the_contact_managerment_system ${source_files})