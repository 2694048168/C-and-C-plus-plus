cmake_minimum_required(VERSION 3.10)

project(DisplayImage)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DisplayImage display_image.cpp)

target_link_libraries(DisplayImage ${OpenCV_LIBS})