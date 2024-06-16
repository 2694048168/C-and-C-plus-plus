cmake_minimum_required(VERSION 3.25)

project("test_labels"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate basic CMake usage"
  LANGUAGES CXX
)

enable_testing()
add_executable(test_labels)
target_sources(test_labels
  PRIVATE src/main.cpp
)

# Label the test test `labeled_test_1 with the label "example"
add_test(NAME labeled_test_1 COMMAND test_labels)
set_tests_properties(labeled_test_1 PROPERTIES LABELS "example")

# Label the test test `labeled_test_2 with the label "will_fail" and mark
# that this test is expected to return a non-zero value on success
add_test(NAME labeled_test_2 COMMAND test_labels extra_arg)
set_tests_properties(labeled_test_2 PROPERTIES LABELS "will_fail" WILL_FAIL True)

# Label the test test `labeled_test_3 with the label "example" and "will_fail" and mark
# that this test is expected to return a non-zero value on success
add_test(NAME labeled_test_3 COMMAND test_labels extra_arg)
set_tests_properties(labeled_test_3 PROPERTIES LABELS "example;will_fail" WILL_FAIL True)
