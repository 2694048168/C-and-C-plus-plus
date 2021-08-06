cmake_minimum_required(VERSION 3.10)

project(pointer_scanning)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pointer_scanning pointer_scanning.cpp)

target_link_libraries(pointer_scanning ${OpenCV_LIBS})