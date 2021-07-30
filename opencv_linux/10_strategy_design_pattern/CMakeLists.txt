cmake_minimum_required(VERSION 3.10)

project(color_dect)

# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 颜色检测库
include_directories(third_party/color_detector)
add_subdirectory(third_party/color_detector)

# 设置所有待编译的源文件
file(GLOB SRC_FILE *.cpp)
message(STATUS ${SRC_FILE})
add_executable(color_dect ${SRC_FILE})

target_link_libraries(color_dect ${OpenCV_LIBS} color_detector_lib)
