cmake_minimum_required(VERSION 3.25)

project("00_Kickstarting_CMake"
    VERSION 2.0
    DESCRIPTION "Simple project for basic CMake usage"
    LANGUAGES CXX
)

message(STATUS [====]${PROJECT_NAME})
message(STATUS [====]${CMAKE_PROJECT_NAME})
message(STATUS [====]${PROJECT_IS_TOP_LEVEL})


add_executable(00_Kickstarting_CMake)
target_sources(00_Kickstarting_CMake
    PRIVATE
    src/main.cpp
)
