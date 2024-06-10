cmake_minimum_required(VERSION 3.25)

project(hello_world_standalone
    VERSION 1.3.5
    DESCRIPTION "A simple Modern C++ project via CMake"
    HOMEPAGE_URL https://github.com/2694048168/C-and-C-plus-plus/tree/master/CMakePractics
    LANGUAGES CXX
)

# set the target name equal project_name
add_executable(${PROJECT_NAME})

# PRIVATE defines that the sources are only used to build this target
# and not for any dependent targets.
# After the scope specifier, there's a list of source files
# that are relative to the path of the current CMakeLists.txt file.
# If needed, the location of the currently processed CMakeLists.txt file
# can be accessed with the "CMAKE_CURRENT_SOURCE_DIR" variable.
message(STATUS [==== current cmake path] ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${PROJECT_NAME}
    PRIVATE

    # ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    src/main.cpp
)

# Adding them with target_sources allows you
# to explicitly define where the sources can be used
# by using PRIVATE, PUBLIC, or INTERFACE.
