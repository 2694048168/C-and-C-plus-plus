cmake_minimum_required(VERSION 3.10)

project(salt_noise)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(salt_noise salt_pepper_noise.cpp)

target_link_libraries(salt_noise ${OpenCV_LIBS})