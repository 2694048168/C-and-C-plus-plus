cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(deriveClass)
target_sources(deriveClass
    PRIVATE
        "derive.cpp"
)

add_executable(virtualClass)
target_sources(virtualClass
    PRIVATE
        "virtual.cpp"
)

add_executable(MatrixDemo)
target_sources(MatrixDemo
    PRIVATE
        "Example/main.cpp"
        "Example/mymatrix.cpp"
        "Example/mymatrix.hpp"
)
