# Top-level CMakeLists file for the whole project

# Specifies the minimum required CMake version to build this project (and
# subprojects) Subprojects may override this value by specifying another version
# requirement in their CMakeLists file (e.g. require a later version of CMake)
cmake_minimum_required(VERSION 3.25)

# The main-whole project
project(
    "CMakePractices"
    VERSION 2.0
    DESCRIPTION "The Modern CMake Best Practices"
    LANGUAGES CXX
)

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

# Variables can be set using the set command and deleted using unset.
# Variable names are case-sensitive
set(MYVAR "WeiLi.Ithaca")
message(STATUS "[====set]The content of MYVAR are ${MYVAR}")
unset(MYVAR)
message(STATUS "[====unset]The content of MYVAR are ${MYVAR}")

# Variables might be scoped in the following way:
# • Function scope: Variables that are set inside a function are only visible inside
# the function.
# • Directory scope: Each of the subdirectories in a source tree binds variables and
# includes any variable bindings from the parent directory.
# • Persistent cache: Cached variables can be either system- or user-defined. These
# persist their values over multiple runs.
# • Passing the PARENT_SCOPE option to set() makes the variable visible in the
# parent scope.
# Variable references can even be nested and are evaluated inside out
set(MANJOR 4)
set(MINOR4 "4.6")
message(STATUS "[====]${MINOR${MANJOR}}")
unset(MANJOR)
unset(MINOR4)

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
