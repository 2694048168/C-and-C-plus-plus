cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()

# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/libs)

# cmake . -B build -G "MinGW Makefiles"
# cmake . -B build -G "Visual Studio 17 2022" -DIS_MinGW=ON
# cmake --build build
option(IS_MinGW "Is using the MinGW Compiler on Windows" OFF)

include_directories(${PROJECT_SOURCE_DIR}/external/GLFW/include)
if(IS_MinGW)
    link_directories(${PROJECT_SOURCE_DIR}/external/GLFW/lib-mingw-w64)
else()
    link_directories(${PROJECT_SOURCE_DIR}/external/GLFW/lib-vc2022)
endif()

add_executable(opengl src/opengl.cpp)
if(IS_MinGW)
    target_link_libraries(opengl
        libglfw3.a
        opengl32.lib
    )
else()
    target_link_libraries(opengl
        glfw3.lib
        opengl32.lib
    )
endif()
