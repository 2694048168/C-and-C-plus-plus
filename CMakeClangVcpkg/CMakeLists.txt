cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# # git clone 'vcpkg' path for the 'vcpkg.cmake'
# set(CMAKE_TOOLCHAIN_FILE "D:/DevelopTools/vcpkg/scripts/buildsystems/vcpkg.cmake")
# # 第三方库的架构(vcpkg.json)
# set(VCPKG_HOST_TRIPLET "x64-windows")
# # 构建目标的架构(可以进行交叉编译了)
# set(VCPKG_TARGET_TRIPLET "x64-windows")

# # 开启 cland 自动补全等功能的文件导出 "build\compile_commands.json"
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)

# message(STATUS "[====]${CMAKE_C_COMPILE}")
# message(STATUS "[====]${CMAKE_CXX_COMPILE}")

# set(CMAKE_C_STANDARD 17)
# set(CMAKE_C_STANDARD_REQUIRED ON)
# # 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_C_EXTENSIONS OFF)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------
project(Modern_CPP
VERSION 0.1.1
DESCRIPTION "the Modern C++ tutorial"
LANGUAGES CXX C
)

# message(STATUS "[====]${CMAKE_BINARY_DIR}")

# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

add_executable(main "main.cpp")

# The package fmt provides CMake targets:
find_package(fmt CONFIG REQUIRED)
target_link_libraries(main PRIVATE fmt::fmt)

# Or use the header-only version
# find_package(fmt CONFIG REQUIRED)
# target_link_libraries(main PRIVATE fmt::fmt-header-only)
