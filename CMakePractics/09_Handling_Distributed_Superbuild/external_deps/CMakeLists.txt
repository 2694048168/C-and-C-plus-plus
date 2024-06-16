cmake_minimum_required(VERSION 3.25)

project(ex01_external_deps
  VERSION 1.0
  DESCRIPTION "External Dependencies in Superbuild"
  LANGUAGES CXX
)

option(USE_SUPERBUILD "Enable superbuild for example 01" OFF)

if(USE_SUPERBUILD)
  include(superbuild.cmake)
else()
  # User decided to not to use superbuild. We have to discover the dependencies
  # from the current environment as usual.
  find_package(GTest 1.10.0 REQUIRED)
  find_package(benchmark 1.6.1 REQUIRED)
endif()

# A simple unit test target
add_executable(ex01_tests)
target_sources(ex01_tests
  PRIVATE src/tests.cpp
)
target_link_libraries(ex01_tests
  PRIVATE GTest::Main
)

# A simple microbenchmark target
add_executable(ex01_benchmarks)
target_sources(ex01_benchmarks
  PRIVATE src/benchmarks.cpp
)
target_link_libraries(ex01_benchmarks
  PRIVATE benchmark::benchmark
)
