cmake_minimum_required(VERSION 3.25)

project("fixture_sample"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate how test fixtures can be run."
  LANGUAGES CXX
)

include(CTest)

# This executable will be the unit under test
add_executable(fixture_sample)
target_sources(fixture_sample
  PRIVATE src/main.cpp
)

# define an executable as fixture
add_executable(fixture_server)
target_sources(fixture_server
  PRIVATE src/server.cpp
)

# The comand to start the server is added as a test and then marked as
# setup function for the fixture named ch7_server
add_test(NAME start_server COMMAND fixture_server)
set_tests_properties(start_server PROPERTIES FIXTURES_SETUP _server)

# The command to stop the server is added as a test and then marked as a
# cleanup function for the fixture named ch7_server
add_test(NAME stop_server COMMAND fixture_server --stop)
set_tests_properties(stop_server PROPERTIES FIXTURES_CLEANUP _server)

# Define the test for the unit under test and tell CMake
# that the fixture 'ch7_server' is required to run it
add_test(NAME ch7_fixture_test COMMAND fixture_sample)
set_tests_properties(ch7_fixture_test PROPERTIES FIXTURES_REQUIRED _server)
