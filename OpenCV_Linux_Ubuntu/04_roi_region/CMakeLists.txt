cmake_minimum_required(VERSION 3.10)

project(roi_region)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(roi_region roi_region.cpp)

target_link_libraries(roi_region ${OpenCV_LIBS})