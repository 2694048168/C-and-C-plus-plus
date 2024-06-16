cmake_minimum_required(VERSION 3.25)

project("test_timeout_example"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate basic CMake usage"
  LANGUAGES CXX
)

add_executable(test_timeout_example)
target_sources(test_timeout_example
  PRIVATE src/main.cpp
)
target_compile_features(test_timeout_example
  PRIVATE cxx_std_17
)

# Mark the test timeout_test with a timeout of 10 seconds
# if the test does not complete by then, it is killed and marked as failing
add_test(NAME timeout_test COMMAND test_timeout_example)
set_tests_properties(timeout_test PROPERTIES TIMEOUT 10)
