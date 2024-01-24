cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(problem_77)

set(LIBS_PATH "${CMAKE_SOURCE_DIR}/PDFWriter")

include_directories(${LIBS_PATH}/)
include_directories(${LIBS_PATH}/FreeType/include)

add_subdirectory(${LIBS_PATH}/PDFWriter)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ZipLib)
target_link_libraries(${PROJECT_NAME} PDFWriter)
target_link_libraries(${PROJECT_NAME} FreeType)
target_link_libraries(${PROJECT_NAME} LibAesgm)
target_link_libraries(${PROJECT_NAME} LibJpeg)
target_link_libraries(${PROJECT_NAME} LibPng)
target_link_libraries(${PROJECT_NAME} LibTiff)
target_link_libraries(${PROJECT_NAME} Zlib)
target_compile_options(${PROJECT_NAME}
    PUBLIC 
        "-std=c++20"
)
