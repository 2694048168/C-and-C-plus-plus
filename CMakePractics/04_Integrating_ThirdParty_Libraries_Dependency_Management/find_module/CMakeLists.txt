cmake_minimum_required(VERSION 3.25)

project(find_module
    VERSION 1.0
    DESCRIPTION "To demonstrate how find-modules work"
    LANGUAGES CXX
)

# The following example will look for a pre-built library in the folder ${CMAKE_CURRENT_BINARY_DIR}/dep. 
# To use it build the project located in `dep_source`
# first and install it in the directory `dep` of the current source treecd

# Tell CMake where to look for find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# try to find the package obscure see the file cmake/FindObscure.cmake for the innner workings
find_package(Obscure QUIET)
# find_package(Obscure REQUIRED)

if(NOT Obscure_FOUND)
    message(STATUS "Prebuilt library 'obscure' not found, skipping find module example")
    return()
endif()

# Create a target to build an executable
add_executable(find_module)
target_sources(find_module
    PRIVATE src/main.cpp
)
target_link_libraries(find_module
    PRIVATE Obscure::Obscure
)
target_compile_features(find_module
    PRIVATE cxx_std_17
)
