cmake_minimum_required(VERSION 3.25)

project("pass_fail_criteria"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate basic CMake usage"
  LANGUAGES CXX
)
add_executable(pass_fail_criteria)
target_sources(pass_fail_criteria
  PRIVATE src/main.cpp
)

# Define a test and define that instead of the return value the output of
# stdout should be parsed to determine if the test passes or fails
# if the output contains either 'Success' or 'success' then the test passes
# if the output contains "Warning", "warning", "error" or "Error" then the test fails
add_test(NAME pass_fail_test COMMAND pass_fail_criteria ARG1)
set_tests_properties(pass_fail_test PROPERTIES
  FAIL_REGULAR_EXPRESSION "[W|w]arning|[E|e]rror"
  PASS_REGULAR_EXPRESSION "[S|s]uccess")
