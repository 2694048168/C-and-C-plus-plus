cmake_minimum_required(VERSION 3.25)

project("simple_test"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate basic CMake usage"
  LANGUAGES CXX
)

# Enable testing for this project
enable_testing()

# Ad the executalbe to test
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
  PRIVATE src/main.cpp
)

# Add a test called example_test which executes the command `simple_test`
add_test(NAME example_test COMMAND ${PROJECT_NAME})

# Add a test called example_test_2 which 
# executes the command `simple_test` with the argument `arg1`
add_test(NAME example_test_2 COMMAND ${PROJECT_NAME} arg1)
