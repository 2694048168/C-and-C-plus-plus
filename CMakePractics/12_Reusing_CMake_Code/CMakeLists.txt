cmake_minimum_required(VERSION 3.25)

project(Reusing_CMake_Code
    VERSION 1.0
    DESCRIPTION
    "Examples of the CMake Best Practices book"
    LANGUAGES CXX
)

add_subdirectory(git_utility)
add_subdirectory(envfile_utility)
add_subdirectory(hadouken)
