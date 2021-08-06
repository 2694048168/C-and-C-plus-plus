cmake_minimum_required(VERSION 3.10)

# project(histogram)

# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置所有待编译的源文件
# file(GLOB SRC_FILE *.cpp)
# message(STATUS ${SRC_FILE})
# add_executable(histogram ${SRC_FILE})

add_executable(histograms histograms.cpp)
add_executable(contentfinder contentfinder.cpp)
add_executable(finder finder.cpp)
add_executable(retrieve retrieve.cpp)
add_executable(integral integral.cpp)
add_executable(tracking tracking.cpp)

target_link_libraries(histograms ${OpenCV_LIBS})
target_link_libraries(contentfinder ${OpenCV_LIBS})
target_link_libraries(finder ${OpenCV_LIBS})
target_link_libraries(retrieve ${OpenCV_LIBS})
target_link_libraries(integral ${OpenCV_LIBS})
target_link_libraries(tracking ${OpenCV_LIBS})
