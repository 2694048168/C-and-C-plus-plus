# example to illustrate unity builds in batch mode
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(unity_build
    VERSION 1.0
    DESCRIPTION
    "A simple C++ project to demonstrate unity-builds"
    LANGUAGES CXX
)

# Create an executable target
add_executable(unity_build)
target_compile_features(unity_build
    PRIVATE cxx_std_17
)

target_sources(unity_build
    PRIVATE
    src/main.cpp
    src/fibonacci.cpp
    src/eratosthenes.cpp
)

# Enable unity build for the target, by default BATCH mode is used
set_target_properties(unity_build
    PROPERTIES UNITY_BUILD True
)

# exclude eratosthenes.cpp from the unity build
set_source_files_properties(src/eratosthenes.cpp PROPERTIES
    SKIP_UNITY_BUILD_INCLUSION YES
)
