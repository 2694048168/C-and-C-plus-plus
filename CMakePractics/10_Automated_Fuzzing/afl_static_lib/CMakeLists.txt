# Static library fuzzing with AFL++
#
# SPDX-License-Identifier: MIT

project(afl_static_lib
  VERSION 1.0
  DESCRIPTION "fuzzing of a static library with AFL++"
  LANGUAGES CXX
)

# Define an executable target named `afl_static_lib`
add_library(afl_static_lib STATIC)

# Specify source files for target named `afl_static_lib`
target_sources(afl_static_lib PRIVATE src/uri_helper.cpp)

# Specify the include directories for the target named `afl_static_lib`
target_include_directories(afl_static_lib PUBLIC include)

# Request compile features for target named `afl_static_lib`.
# Requested `cxx_std_11` feature determines the minimum C++ standard required
# to build this target. It's PRIVATE, since it is a build-time requirement only.
target_compile_features(afl_static_lib PRIVATE cxx_std_11)

# Include the fuzzing target
add_subdirectory(fuzz)
