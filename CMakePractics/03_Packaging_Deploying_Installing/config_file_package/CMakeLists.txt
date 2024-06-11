cmake_minimum_required(VERSION 3.25)

project(config_file_package
    VERSION 1.0.1
    DESCRIPTION "Config-file package via CMake"
    LANGUAGES CXX
)

# Defines the ${CMAKE_INSTALL_INCLUDEDIR} variable.
include(GNUInstallDirs)

# This is the default install directory for config-file package cmake files
# It is relative to install prefix.
set(lib_INSTALL_CMAKEDIR
    cmake CACHE PATH "Installation directory for config-file package cmake files"
)

# Define a static library target named `config_file_package`
add_library(config_file_package STATIC)

# Specify source files for target named `config_file_package`
target_sources(config_file_package
    PRIVATE src/hello.cpp
)

# Specify the include directories for the target named `config_file_package`
target_include_directories(config_file_package
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Request compile features for target named `config_file_package`.
# Requested `cxx_std_17` feature determines the minimum C++ standard required
# to build this target. It's PRIVATE, since it is a build-time requirement only.
target_compile_features(config_file_package
    PUBLIC cxx_std_17
)

# Make executable target `config_file_package` installable.
install(TARGETS config_file_package
    EXPORT lib_export # Associate installed target files of the targets in TARGETS parameter with export name of lib_export

    # Note that this does not export anything yet, it just defines a name and context for the exports. The install(EXPORT)
    # call does the actual target exporting.
    # Will be appended to INTERFACE_INCLUDE_DIRECTORIES of exported targets.
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the header files. Since header files are not listed as output artifacts, they have
# to be installed separately.
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate the required import code for the content in <export name>
# into lib-config.cmake CMake file. Prefix all target names with
# lib:: while doing so (NAMESPACE). DESTINATION argument recommended to be a path
# that will be picked by find_package() by default ('cmake' in our example).
install(EXPORT lib_export
    FILE lib-config.cmake
    NAMESPACE lib::
    DESTINATION ${lib_INSTALL_CMAKEDIR}
)

# Defines write_basic_package_version_file
include(CMakePackageConfigHelpers)

# Create a package version file for the package. This will allow consumers to specify
# version argument to select between several versions of the package via specifying VERSION argument
# to find_package().
# We're not explicitly specifying the VERSION parameter to allow `write_basic_package_version_file` to
# get it from project(...) version (which is 1.0.1)
write_basic_package_version_file(
    "lib-config-version.cmake"

    # Package compatibility strategy. SameMajorVersion is essentially `semantic versioning`.
    COMPATIBILITY SameMajorVersion
)

# Install command for deploying Config-file package files into the target system. It must be present
# in the same directory as `lib-config.cmake` file.
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lib-config-version.cmake"
    DESTINATION "${lib_INSTALL_CMAKEDIR}"
)
