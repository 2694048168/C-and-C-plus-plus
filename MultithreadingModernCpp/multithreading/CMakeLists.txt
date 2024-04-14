# 设置 CMake 所需要的最小版本
cmake_minimum_required(VERSION 3.20.0)

# 配置项目名称(solution in VS)以及简要信息
project(Multithreading
    VERSION 0.0.1
    DESCRIPTION "The basic Multithreading example of C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C
)

# MinGW+Clang | Ninja | CMake | VSCode + clangd + c/c++ Snippets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 18)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 条件判断 Debug & Release模式下的编译指令
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# -------------------------------------------
add_executable(${PROJECT_NAME} basic_thread.cpp)
