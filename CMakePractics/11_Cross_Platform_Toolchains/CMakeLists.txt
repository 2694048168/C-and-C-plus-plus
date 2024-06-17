cmake_minimum_required(VERSION 3.25)

project("Cross_Platform_Toolchains"
    VERSION 1.0
    DESCRIPTION
    "A collection of sample C++ projects to demonstrate cross-compiling, toolchains and toolchain checks"
    LANGUAGES CXX
)

# The examples ion android-example and iOS_example are to be used standalone with the
# the respective SDKs and NDKs
add_subdirectory(check_source_compiles)
add_subdirectory(check_push_pop)
