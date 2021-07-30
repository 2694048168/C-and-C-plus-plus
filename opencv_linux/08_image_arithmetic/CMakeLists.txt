cmake_minimum_required(VERSION 3.10)

project(image_arithmetic)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image_arithmetic image_add_operator.cpp)

target_link_libraries(image_arithmetic ${OpenCV_LIBS})