cmake_minimum_required(VERSION 3.25)

project(ImageConvert
VERSION 0.1.1
DESCRIPTION "the image format between OpenCV and Qt"
LANGUAGES CXX
)

# 开启 cland 自动补全等功能的文件导出 "build\compile_commands.json"
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_STANDARD 18)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# =========== thirdparty library ===========
set(OpenCV_DIR "D:/DevelopTools/OpenCV/")
# set(OpenCV_DIR "D:/DevelopTools/OpenCV_MinGW/")
# find_package(OpenCV REQUIRED)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
# message(STATUS "OpenCV library status:")
# message(STATUS "    config: ${OpenCV_DIR}")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
# include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OpenCV_LIBS})

include_directories("${OpenCV_DIR}/include/")
link_directories("${OpenCV_DIR}/lib/")

# set(Qt6_DIR "D:/DevelopTools/QT/6.2.4/msvc2019_64/lib/cmake/Qt6/")
# set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/6.2.4/mingw_64/")
set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/6.2.4/msvc2019_64/")
# find_package(Qt6 REQUIRED COMPONENTS Widgets)
# find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui)
# qt_standard_project_setup()

# Set your CMAKE_PREFIX_PATH environment variable to the Qt 5 installation prefix.
# set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/5.15.2/msvc2019_64/")
# Set the Qt5_DIR in the CMake cache to the location of the Qt5Config.cmake file.
# set(Qt5_DIR "D:/DevelopTools/QT/5.15.2/msvc2019_64/lib/cmake/Qt5/")
# find_package(Qt5 COMPONENTS Widgets REQUIRED)
# find_package(Qt5 COMPONENTS Core REQUIRED)
# find_package(Qt5 COMPONENTS Gui REQUIRED)

add_executable(main
    "main.cpp"
    "utility.hpp"
    "utility.cpp"
)
target_link_libraries(main
    PRIVATE
        # "${OpenCV_LIBS}"
        # "libopencv_imgcodecs455.dll.a"
        "opencv_world480d.lib"
        # "Qt6::Widgets"
        # "Qt6::Core"
        "Qt6::Gui"
        # "Qt5::Widgets"
        # "Qt5::Core"
        # "Qt5::Gui"
)

# 把需要动态库拷贝到可执行路径下
file(GLOB LIB_DLL
    # "D:/DevelopTools/QT/5.15.2/msvc2019_64/bin/Qt5Cored.dll"
    # "D:/DevelopTools/QT/5.15.2/msvc2019_64/bin/Qt5Guid.dll"
    # "D:/DevelopTools/QT/5.15.2/msvc2019_64/bin/Qt5Widgetsd.dll"
    "D:/DevelopTools/QT/6.2.4/msvc2019_64/bin/Qt6Cored.dll"
    "D:/DevelopTools/QT/6.2.4/msvc2019_64/bin/Qt6Guid.dll"
    "D:/DevelopTools/QT/6.2.4/msvc2019_64/bin/Qt6Widgetsd.dll"
    "${OpenCV_DIR}/bin/opencv_world480d.dll"
)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${LIB_DLL} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    # file(COPY ${LIB_DLL} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
else()
    file(COPY ${LIB_DLL} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()
