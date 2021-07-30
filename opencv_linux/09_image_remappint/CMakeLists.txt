cmake_minimum_required(VERSION 3.10)

project(remapping_image)

# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置所有待编译的源文件
file(GLOB SRC_FILE *.cpp)
message(STATUS ${SRC_FILE})
add_executable(remapping_image ${SRC_FILE})

target_link_libraries(remapping_image ${OpenCV_LIBS})


# # ----------------------------------------------
# # 自己项目中的使用 第三方(自己编写的模块用法
# # 通过设置编译时候需要的头文件和链接时候需要的库文件路径
# # ----------------------------------------------
# cmake_minimum_required(VERSION 3.10)

# project(remapping_image)

# # 设置 OpenCV 头文件查找路径
# include_directories("../third_party/include/")
# # 添加 第三块模块 里面的 CMakeLists.txt
# add_subdirectory(third_party/third_module)

# # 设置所有待编译的源文件
# file(GLOB SRC_FILE *.cpp)
# message(STATUS ${SRC_FILE})
# add_executable(remapping_image ${SRC_FILE})

# # 设置 OpenCV 库文件路径
# file(GLOB LIB_FILE "../third_party/lib")
# target_link_libraries(remapping_image ${LIB_FILE})
# # -----------------------------------------
