cmake_minimum_required(VERSION 3.27.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CXX_STANDARD 17)
set(CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

project(problem_09)

add_executable(problem_09 main.cpp)
