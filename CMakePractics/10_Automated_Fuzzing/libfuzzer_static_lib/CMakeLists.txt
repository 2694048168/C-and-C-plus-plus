# Static library fuzzing with libfuzzer
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(libfuzzer_static
  VERSION 1.0
  DESCRIPTION "fuzzing of a static library with libfuzzer"
  LANGUAGES CXX
)

# Define an executable target named `libfuzzer_static`
add_library(libfuzzer_static STATIC)

# Specify source files for target named `libfuzzer_static`
target_sources(libfuzzer_static
  PRIVATE src/library.cpp
)

# Specify the include directories for the target named `libfuzzer_static`
target_include_directories(libfuzzer_static
  PUBLIC include
)

# Request compile features for target named `libfuzzer_static`.
# Requested `cxx_std_17` feature determines the minimum C++ standard required
# to build this target. It's PRIVATE, since it is a build-time requirement only.
target_compile_features(libfuzzer_static
  PRIVATE cxx_std_17
)

# Include the fuzzing target
add_subdirectory(fuzz)
