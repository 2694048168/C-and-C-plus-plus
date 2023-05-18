cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# MinGW+Clang | Ninja | CMake | VSCode + clangd + c/c++ Snippets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER clang) # clang | gcc | MSVC(cl)
set(CMAKE_CXX_COMPILER clang++) # clang++ | g++ | | MSVC(cl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_EXTENSIONS OFF)

# .sln
project(CMake_Tutorial
    VERSION 2.1.1
    DESCRIPTION "the CMake tutorial"
    LANGUAGES CXX
)

set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/calc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

# graphviz to visual the relationship
add_custom_target(graphviz ALL
    COMMAND ${CMAKE_COMMAND} "--graphviz=visual_tree.dot" .
    COMMAND dot -Tpng visual_tree.dot -o visual_tree.png
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
