cmake_minimum_required(VERSION 3.10)

project(load_display_save_image)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(load_display_save_image load_display_save_image.cpp)

target_link_libraries(load_display_save_image ${OpenCV_LIBS})