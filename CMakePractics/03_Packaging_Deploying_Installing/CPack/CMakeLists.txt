cmake_minimum_required(VERSION 3.25)

project(pack_cmake
    VERSION 1.0
    DESCRIPTION "Packaging with CPack"
    LANGUAGES CXX
)

if(NOT PROJECT_IS_TOP_LEVEL)
    message(FATAL_ERROR "pack_cmake project is intended to be a standalone, top-level project. Do not include this directory.")
endif()

add_subdirectory(executable)
add_subdirectory(library)

# We will not explicitly specify project name and version here and
# let CPack to get project name and version from the project()
set(CPACK_PACKAGE_VENDOR "CBP Authors")

# Enable DEB, RPM TBZ2 and ZIP generators by default
set(CPACK_GENERATOR "DEB;RPM;TBZ2;ZIP")

# Use all available cores when parallelism is supported
set(CPACK_THREADS 0)

# The DEB generator requires CPACK_DEBIAN_PACKAGE_MAINTAINER
# value to be set.
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "CBP Authors")

# Enable packaging support for the project
include(CPack)
