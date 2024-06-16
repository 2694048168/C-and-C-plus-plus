# Driver application
#
# SPDX-License-Identifier: MIT

project(afl_fuzz
  VERSION 1.0
  DESCRIPTION "fuzzing of a static library with AFL++, driver application"
  LANGUAGES CXX
)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "afl_fuzz requires GNU compiler. example will not be built.")
  return()
endif()

# Define an executable target named `afl_fuzz`
add_executable(afl_fuzz)

# Specify source files for target named `afl_fuzz`
target_sources(afl_fuzz PRIVATE fuzz_uri_helper.cpp)

# Request compile features for target named `afl_fuzz`.
# Requested `cxx_std_11` feature determines the minimum C++ standard required
# to build this target. It's PRIVATE, since it is a build-time requirement only.
target_compile_features(afl_fuzz PRIVATE cxx_std_11)

# link against the fuzzing target
target_link_libraries(afl_fuzz PRIVATE afl_static_lib)
