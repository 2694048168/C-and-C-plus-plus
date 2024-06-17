# example to use with the emulator when cross compiled
# This example is supposed to be compiled with the `arm64.toolchain.cmake` toolchain
# in the same directory, which defines the qemu command to run the test inside the emulator
#
# check the github action workflow cross-compiling.yml
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(emulator_example
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate basic CMake usage"
  LANGUAGES CXX
)

# enable testing
include(CTest)

# define an executable target
add_executable(emulator_example)
target_sources(emulator_example
  PRIVATE src/main.cpp
)

add_test(NAME emulator_test COMMAND emulator_example)
