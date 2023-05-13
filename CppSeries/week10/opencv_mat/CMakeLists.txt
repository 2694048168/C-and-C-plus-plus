cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 11)

project(OpenCV_MatExample)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(OpenCV_MatExample "mat_example.cpp")
target_link_libraries(OpenCV_MatExample
    PRIVATE
        ${OpenCV_LIBS}
)