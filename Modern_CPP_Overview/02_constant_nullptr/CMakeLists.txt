cmake_minimum_required(VERSION 3.10)

# 指定编译器
# set(CMAKE_CXX_COMPILER "D:/mingw64/bin/g++.exe")
set(CMAKE_CXX_COMPILER "D:/mingw64/bin/clang++.exe")

# 设置编译选项
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++2a")
# 设置编译模式, Debug、Release
set (CMake_BUILD_TYPE "Debug")

project(constant_nullptr)

add_executable(${PROJECT_NAME} constant_nullptr.cpp)