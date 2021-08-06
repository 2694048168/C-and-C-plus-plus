cmake_minimum_required(VERSION 3.10)

# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add executable
add_executable(segment segment.cpp)
add_executable(mserFeatures mserFeatures.cpp)

# link libraries
target_link_libraries(segment ${OpenCV_LIBS})
target_link_libraries(mserFeatures ${OpenCV_LIBS})
