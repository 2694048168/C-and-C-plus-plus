cmake_minimum_required(VERSION 3.25)

project(Superbuild
    VERSION 1.0
    DESCRIPTION
    "Examples for chapter 10 of the CMake Best Practices book"
    LANGUAGES CXX
)

add_subdirectory(external_deps)
add_subdirectory(external_deps_with_extproject)
