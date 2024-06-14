cmake_minimum_required(VERSION 3.25)

project(vcpkg_example
    VERSION 1.0
    DESCRIPTION
    "A simple C++ project to demonstrate retrieving third-party dependencies with vcpkg"
    LANGUAGES CXX
)

find_package(RapidJSON QUIET)

if(NOT RapidJSON_FOUND)
    message(WARNING "RapidJSON not found, skipping vcpkg example")
endif()

add_executable(vcpkg_example)
target_compile_features(vcpkg_example
    PRIVATE cxx_std_17
)
target_sources(vcpkg_example
    PRIVATE src/main.cpp
)
target_link_libraries(vcpkg_example
    PRIVATE rapidjson
)
