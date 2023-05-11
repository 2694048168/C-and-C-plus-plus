cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(hello)
target_sources(hello
    PRIVATE
        "hello.cpp"
)

add_executable(main)
target_sources(main
    PRIVATE
        "main.cpp"
        "arithmetic/mymul.cpp"
        "arithmetic/mymul.hpp"
)

add_executable(integersRange)
target_sources(integersRange
    PRIVATE
        "exercises.cpp"
)
