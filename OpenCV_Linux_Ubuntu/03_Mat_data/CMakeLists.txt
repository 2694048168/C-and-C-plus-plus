cmake_minimum_required(VERSION 3.10)

project(mat_data)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mat_data mat_data.cpp)

target_link_libraries(mat_data ${OpenCV_LIBS})