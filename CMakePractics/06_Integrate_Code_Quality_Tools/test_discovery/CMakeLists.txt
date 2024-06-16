cmake_minimum_required(VERSION 3.25)

project("test_discovery_example"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate test discovery with Catch2 in CMake"
  LANGUAGES CXX
)

# Fetch Catch2 from github and make it available
include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.8)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib/)
include(CTest)
include(Catch)

add_executable(test_discovery_example)
target_sources(test_discovery_example
  PRIVATE src/test.cpp
)
target_link_libraries(test_discovery_example
  PRIVATE Catch2::Catch2
)

# use the discovery function of catch2
catch_discover_tests(test_discovery_example)
