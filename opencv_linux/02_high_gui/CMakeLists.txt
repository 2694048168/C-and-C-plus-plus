cmake_minimum_required(VERSION 3.10)

project(high_gui)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(high_gui high_gui_mouse.cpp)

target_link_libraries(high_gui ${OpenCV_LIBS})