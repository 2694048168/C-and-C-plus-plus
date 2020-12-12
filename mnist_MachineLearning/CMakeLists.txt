cmake_minimum_required(VERSION 3.5)

project(Extract_Transform_Load)

add_compile_options(-Wall -std=c++17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(Extract_Transform_Load extract_transform_load.cpp src/data.cpp src/data_handler.cpp)
