cmake_minimum_required(VERSION 3.25)

project(consumer_cmake_config
    VERSION 1.0
    DESCRIPTION "Out ot project config-file package consumer"
    LANGUAGES CXX
)

if(NOT PROJECT_IS_TOP_LEVEL)
    message(FATAL_ERROR "consumer project is intended to be a standalone, top-level project. Do not vendor this directory.")
endif()

# #################################################################################
# IMPORTANT NOTE: If "lib" is installed on a non-default install prefix
# this project must be configured with -DCMAKE_PREFIX_PATH:STRING=<install prefix>
# variable. This enables find_package() to discover the package even though it is
# installed to a custom prefix.
# #################################################################################
set(CMAKE_PREFIX_PATH external)

# Let's find our package. We're explicitly specifying CONFIG argument
# to not to search for MODULE packages. CMake will fail if package is not present
# since REQUIRED argument is given. Also, we're explicitly requesting major version
# of 1. We are able to do this since we've provided a <project>-config-version.cmake file too.
# The packages with no -config-version.cmake file are not considered by find_package()
# when version is specified.
# -----find_package(…) must only consider CONFIG packages
# -----and packages specified in this find_package(…) call are required.
find_package(lib 1 CONFIG REQUIRED)

# Define the executable that will consume the "lib" package
add_executable(${PROJECT_NAME}
    src/main.cpp
)

# Set the required C++ standard for the target
target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_17
)

# Link executable to "lib" package found by find_package() before. Target is under
# `lib::` namespace since export is made using that namespace.
target_link_libraries(${PROJECT_NAME} PRIVATE lib::config_file_package)
