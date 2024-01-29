cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(problem_91)

include_directories(${LIBS_PATH}/cryptopp)
add_definitions(-DCRYPTOPP_CXX11_LAMBDA)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    PUBLIC 
        "cryptlib"
)
target_compile_options(${PROJECT_NAME}
    PUBLIC 
        "-std=c++20"
)
set_target_properties (${PROJECT_NAME}
    PROPERTIES FOLDER "11.Cryptography"
)