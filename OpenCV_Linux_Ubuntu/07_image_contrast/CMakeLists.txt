cmake_minimum_required(VERSION 3.10)

project(sharpen_image)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sharpen_image image_contrast.cpp)

target_link_libraries(sharpen_image ${OpenCV_LIBS})