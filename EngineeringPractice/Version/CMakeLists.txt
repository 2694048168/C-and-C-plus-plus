cmake_minimum_required(VERSION 3.25.0)

project(Version_Test
    VERSION 1.0.1
    DESCRIPTION "The Models C++ Parallel and Concurrency"
    HOMEPAGE_URL "https://github.com/2694048168/"
    LANGUAGES C CXX
 )
 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_custom_target(
    RunPowerShellScript ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Running PowerShell Script..."
    COMMAND powershell -ExecutionPolicy Bypass -File ${CMAKE_SOURCE_DIR}/Version.ps1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Executing PowerShell Script to generate Software Code Information"
)

add_executable(${PROJECT_NAME} test_main.cpp)
add_executable(${PROJECT_NAME}_1 test_datetime_main.cpp)
add_executable(${PROJECT_NAME}_2 test_semver_main.cpp semver.cpp)
