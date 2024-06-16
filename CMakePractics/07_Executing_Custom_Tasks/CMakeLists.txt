cmake_minimum_required(VERSION 3.25)

project("Modern CMake Best Practices Custom"
  VERSION 1.0
  DESCRIPTION
  "A collection of sample C++ applications and libraries to demonstrate usage of custom commands and custom targets"
  LANGUAGES CXX
)

add_subdirectory(custom_target)
add_subdirectory(custom_command)
add_subdirectory(create_source_file)
add_subdirectory(chaining_custom_commands)
add_subdirectory(custom_target_dependency)
add_subdirectory(execute_process)
add_subdirectory(configure_file)
