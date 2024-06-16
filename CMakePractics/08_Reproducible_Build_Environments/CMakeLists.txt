cmake_minimum_required(VERSION 3.25)

project(simple_executable_docker
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate basic CMake usage"
  LANGUAGES CXX
)

add_executable(simple_docker
  src/main.cc
)
