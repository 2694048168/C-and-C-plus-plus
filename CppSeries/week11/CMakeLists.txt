cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(StringExample1)
target_sources(StringExample1
    PRIVATE
        "example1/main.cpp"
        "example1/mystring.hpp"
)

add_executable(StringExample2)
target_sources(StringExample2
    PRIVATE
        "example2/main.cpp"
        "example2/mystring.hpp"
)

add_executable(sharedPtr)
target_sources(sharedPtr
    PRIVATE
        "shared_ptr.cpp"
)

add_executable(uniquePtr)
target_sources(uniquePtr
    PRIVATE
        "unique_ptr.cpp"
)

add_executable(labMatrix)
target_sources(labMatrix
    PRIVATE
        "matrix_lab.cpp"
)
