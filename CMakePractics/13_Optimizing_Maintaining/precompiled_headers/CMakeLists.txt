# example to illustrate the creation and use of precompiled headers
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(precompiled_headers
    VERSION 1.0
    DESCRIPTION
    "A simple C++ project to demonstrate creating precompiled headers"
    LANGUAGES CXX
)

# Create an executable target
add_executable(precompiled_headers)

# Add sources to the executable target
target_sources(precompiled_headers
    PRIVATE
    src/main.cpp
    src/fibonacci.cpp
    src/eratosthenes.cpp
    src/pythagoras.cpp
)

# Add headers to the generated precompiled header for target precompiled_headers
target_precompile_headers(precompiled_headers PRIVATE
    src/fibonacci.h <cstdint> <vector> src/eratosthenes.h
)
