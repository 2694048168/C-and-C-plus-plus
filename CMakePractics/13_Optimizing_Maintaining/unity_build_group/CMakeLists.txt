# example to illustrate group mode for unity builds
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(unity_build_group
    VERSION 1.0
    DESCRIPTION
    "A simple C++ project to demonstrate unity build group modes"
    LANGUAGES CXX
)

add_executable(unity_build_group)

target_sources(unity_build_group
    PRIVATE
    src/main.cpp
    src/fibonacci.cpp
    src/eratosthenes.cpp
    src/pythagoras.cpp
)
target_compile_features(unity_build_group
    PRIVATE cxx_std_17
)

# Enable unity build for the target
# by default BATCH mode is used for unity builds, so the build mode is changed to GROUP
set_target_properties(unity_build_group
    PROPERTIES
    UNITY_BUILD True
    UNITY_BUILD_MODE GROUP
)

# Add main.cpp and fibonacci.cpp to the unuty group1
set_source_files_properties(src/main.cpp src/fibonacci.cpp PROPERTIES UNITY_GROUP group1)

# add erathostenes.cpp and pythagoras.cpp to group2
set_source_files_properties(src/erathostenes.cpp src/pythagoras.cpp PROPERTIES UNITY_GROUP group2)
