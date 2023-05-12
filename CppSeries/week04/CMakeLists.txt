cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(array)
target_sources(array
    PRIVATE
        "array.cpp"
)

add_executable(stdstring)
target_sources(stdstring
    PRIVATE
        "stdstring.cpp"
)

add_executable(struct)
target_sources(struct
    PRIVATE
        "struct.cpp"
)

add_executable(union)
target_sources(union
    PRIVATE
        "union.cpp"
)

add_executable(enum)
target_sources(enum
    PRIVATE
        "enum.cpp"
)

add_executable(multipleFile)
target_sources(multipleFile
    PRIVATE
        "src/main.cpp"
        "src/printhello.cpp"
        "src/factorial.cpp"
        "src/function.hpp"
)
