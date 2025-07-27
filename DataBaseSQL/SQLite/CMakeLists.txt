cmake_minimum_required(VERSION 3.20.0)

project(SQLiteBasic
    VERSION 0.0.1
    DESCRIPTION "The Modern C++ SQLite database Project"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 条件判断编译模式 Debug or Release
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
    message(STATUS "====Debug mode: ${CMAKE_CXX_FLAGS_DEBUG}")
    set(Build_Mode "Debug")
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
    message(STATUS "====Release mode: ${CMAKE_CXX_FLAGS_RELEASE}")
    set(Build_Mode "Release")
else()
    message(STATUS "====Build mode: ${CMAKE_BUILD_TYPE}")
    message(STATUS "====: ${CMAKE_CXX_FLAGS_RELEASE}")
    set(Build_Mode "")
endif()

message(STATUS "======== Build Mode: ${Build_Mode}")

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 以下两条语句分别设置了Debug版本和Release版本可执行文件的输出目录,
# 一旦设置上面的属性,在任何环境下生成的可执行文件都将直接放在你所设置的目录.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)

# 上面两条语句分别设置了Debug版本和Release版本库文件的输出目录,
# 一旦设置上面的属性,在任何环境下生成的库文件都将直接放在你所设置的目录.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)

# 头文件路径
set(Include_Dir ${CMAKE_SOURCE_DIR}/external/)
# list(APPEND Include_Dir ${CMAKE_SOURCE_DIR}/src/)
message(STATUS "=====Include_Dir: ${Include_Dir}")

# 链接库文件路径
set(Link_Dir ${CMAKE_SOURCE_DIR}/external/)
# list(APPEND Link_Dir ${CMAKE_SOURCE_DIR}/external/log4cplus/)
# list(APPEND Link_Dir ${CMAKE_SOURCE_DIR}/external/spdlog/)
list(APPEND Link_Dir ${LIBRARY_OUTPUT_PATH})
message(STATUS "=====Link_Dir: ${Link_Dir}")

# -------------------------------------------
# external 第三方库

# 添加 CMake 子目录, 寻找该目录下的 CMakeLists.txt
# add_subdirectory(src/logger)

add_executable(${PROJECT_NAME} mainBasic.cpp)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${Include_Dir}
)
target_link_directories(${PROJECT_NAME}
    PUBLIC
        ${Link_Dir}
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        sqlite3.lib
)
