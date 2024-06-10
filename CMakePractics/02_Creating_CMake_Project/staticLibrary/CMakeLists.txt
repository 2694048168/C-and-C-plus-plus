cmake_minimum_required(VERSION 3.25)

project(hello_lib
    VERSION 1.2.0
    DESCRIPTION "A simple C++ project to demonstrate creating libraries in CMake"
    LANGUAGES CXX
)

# pass "STATIC" or "SHARED" instead to determine
# the linking type of the library explicitly
add_library(${PROJECT_NAME} STATIC)

# If the type of the library is omitted, the "BUILD_SHARED_LIBS" variable determines
# whether the libraries are built as shared or static libraries by default.
message(STATUS [====the var value: ]${BUILD_SHARED_LIBS})

# The "PRIVATE" and "PUBLIC" keywords specify where the sources should be used
# for compiling. Specifying "PRIVATE" means that the sources will only be used
# in the target hello itself. If "PUBLIC" is used, then the sources will be
# added to hello and any target that links to hello.
# The "INTERFACE" keyword would mean that the sources are not added to hello
# but should be added to anything that links against hello.
target_sources(${PROJECT_NAME}
    PRIVATE
    src/hello.cpp
    src/internal.cpp
)

# Since it is used to compile the library itself and to interface against the library,
# it is set to PUBLIC.
target_compile_features(${PROJECT_NAME}
    PUBLIC
    cxx_std_17
)

# All the files inside the folders specified by
# this command can be accessed using #include <file.hpp> (with the angle brackets)
# instead of #include "", although the version with the quotes may still work.
# *PRIVATE includes paths that will not be included in the target property;
# that is, INTERFACE_INCLUDE_DIRECTORIES.
target_include_directories(${PROJECT_NAME}
    PRIVATE src/hello
    PUBLIC include
)
