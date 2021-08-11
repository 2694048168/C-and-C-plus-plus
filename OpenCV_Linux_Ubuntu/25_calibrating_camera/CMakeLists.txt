# cmake needs this line
cmake_minimum_required(VERSION 3.10)

# Define project name
project(calibrate)

# set(OpenCV_DIR /home/weili/opencv4-install/)
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
# CMake 链接第三方库方法，通过 /usr/local/lib/cmake/opencv4/*.cmake 文件
# 其中包含 OpenCV 第三方库的头文件位置以及库文件，直接查找即可 find_package

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable( calibrate calibrate.cpp CameraCalibrator.cpp)
add_executable( cameraPose camera_pose.cpp)
add_executable( estimateE estimateE.cpp triangulate.cpp)
add_executable( stereoMatcher stereoMatcher.cpp)

# Link your application with OpenCV libraries
target_link_libraries( calibrate ${OpenCV_LIBS})
target_link_libraries( cameraPose ${OpenCV_LIBS})
target_link_libraries( estimateE ${OpenCV_LIBS})
target_link_libraries( stereoMatcher ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/../images/bench2.jpg
            ${CMAKE_SOURCE_DIR}/../images/soup1.jpg
            ${CMAKE_SOURCE_DIR}/../images/soup2.jpg
            ${CMAKE_SOURCE_DIR}/../images/brebeuf1.jpg
            ${CMAKE_SOURCE_DIR}/../images/brebeuf2.jpg			
            calib.xml)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
FILE(COPY ${CMAKE_SOURCE_DIR}/../images/chessboards/ DESTINATION ./chessboards/)
FILE(COPY ${CMAKE_SOURCE_DIR}/../images/chessboards/ DESTINATION "Debug/chessboards/")
FILE(COPY ${CMAKE_SOURCE_DIR}/../images/chessboards/ DESTINATION "Release/chessboards/")