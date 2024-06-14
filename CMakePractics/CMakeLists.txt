# Top-level CMakeLists file for the whole project

# Specifies the minimum required CMake version to build this project (and
# subprojects) Subprojects may override this value by specifying another version
# requirement in their CMakeLists file (e.g. require a later version of CMake)
cmake_minimum_required(VERSION 3.25)

# CMake policies
# For the top-level CMakeLists.txt file, cmake_minimum_required must be called
# before any call to the project as it also sets which internal policies for CMake are used to
# build the project.
# Policies can be disabled or enabled using the cmake_policy call.
# Policies are used to maintain backward compatibility across multiple CMake releases
# configured to use the OLD behavior or as NEW
# https://cmake.org/cmake/help/latest/command/cmake_policy.html
# cmake_policy(SET CMP0121 NEW)
# cmake_policy(SET CMP0121 OLD)

# The main-whole project
project(
    "CMakePractices"
    VERSION 2.0
    DESCRIPTION "The Modern CMake Best Practices"
    LANGUAGES CXX
)

# --------------------------------------------------------------------
# The following are some additional variables regarding the projects. All of them can be
# prefixed with CMAKE_ to the value for the root project. If they're not defined in the
# project() directive, the strings are empty:
# • PROJECT_DESCRIPTION: The description string of the project
# • PROJECT_HOMEPAGE_URL: The URL string for the project
# • PROJECT_VERSION: The full version that's given to the project
# • PROJECT_VERSION_MAJOR: The first number of the version string
# • PROJECT_VERSION_MINOR: The second number of the version string
# • PROJECT_VERSION_PATCH: The third number of the version string
# • PROJECT_VERSION_TWEAK: The fourth number of the version string
message(STATUS [====]${PROJECT_NAME})
message(STATUS [====]${CMAKE_PROJECT_NAME})
message(STATUS [====]${PROJECT_IS_TOP_LEVEL})
message(STATUS [====The version]${CMAKE_PROJECT_VERSION})
message(STATUS [====The major version]${CMAKE_PROJECT_VERSION_MAJOR})
message(STATUS [====The minor version]${CMAKE_PROJECT_VERSION_MINOR})
message(STATUS [====The patch version]${CMAKE_PROJECT_VERSION_PATCH})
message(STATUS [====The tweak version]${CMAKE_PROJECT_VERSION_TWEAK})

# Different toolchains and build types
# A toolchain typically consists of a series of programs that can compile and link
# binaries, as well as creating archives and similar.
# This toolchain is stored in the cache, so if the toolchain
# changes, the cache must be deleted and rebuilt.
# overwrite the C++ compiler to use by passing the respective cmake
# variable using -D
# -D CMAKE_CXX_COMPILER=g++-12
# -D CMAKE_CXX_LINKER

# debug build that contains all debug symbols
# and release builds that are optimized.
# CMake natively provides four build types:
# • Debug: This is non-optimized and contains all the debug symbols. Here, all the
# asserts are enabled. This is the same as setting -O0 -g for GCC and Clang.
# • Release: This is optimized for speed without debugging symbols and asserts
# disabled. Usually, this is the build type that is shipped. This is the same as
# -O3 -DNDEBUG.
# • RelWithDebInfo: This provides optimized code and includes debug symbols
# but disabled asserts, which is the same as -O2 -g -DNDEBUG.
# • MinSizeRel: This is the same as Release but optimized for a small binary size
# instead of speed, which would be -Os -DNDEBUG. Note that this configuration is
# not supported for all generators on all platforms.

# Maintaining good build configurations with presets
include(CTest)

# Set the default build type for single-config generators. The build type
# variable is still overridable from outside.
set(CMAKE_BUILD_TYPE
    "Debug"
    CACHE STRING "Default build type for single-config generators"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Print generator type and build type (if applicable)
get_property(
    is_multi_config_generator GLOBAL
    PROPERTY GENERATOR_IS_MULTI_CONFIG
)

if(NOT is_multi_config_generator)
    message(STATUS "Using a single-config generator (${CMAKE_GENERATOR})")
    message(STATUS "Current build type is `${CMAKE_BUILD_TYPE}`")
else()
    message(STATUS "Using a multi-config generator (${CMAKE_GENERATOR})")
endif()

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Add sub-cmake to root project
add_subdirectory(00_Kickstarting_CMake)
add_subdirectory(01_Accessing_CMake_Best_Ways)
add_subdirectory(02_Creating_CMake_Project)
add_subdirectory(03_Packaging_Deploying_Installing)
add_subdirectory(04_Integrating_ThirdParty_Libraries_Dependency_Management)
