cmake_minimum_required(VERSION 3.25)

project(
    hello_header_only
    VERSION 1.0
    DESCRIPTION "Interface or header only example"
    LANGUAGES CXX
)

# Create a library target
add_library(${PROJECT_NAME} INTERFACE)

# expose the include directories
target_include_directories(${PROJECT_NAME} INTERFACE include/)

# expose the minium C++ standard needed to compile this library
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
