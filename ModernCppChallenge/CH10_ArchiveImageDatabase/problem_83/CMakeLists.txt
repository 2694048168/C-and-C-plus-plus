cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")

project(problem_83)

include_directories(${CMAKE_SOURCE_DIR}/thirdParty/include/)
link_directories(${CMAKE_SOURCE_DIR}/thirdParty/lib/)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        "ZipLib.lib"
        "zlib.lib"
        "lzma.lib"
        "bzip.lib"
)
target_compile_options(${PROJECT_NAME}
    PUBLIC 
        "-std=c++20"
)
