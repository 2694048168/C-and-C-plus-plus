cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(singleDemo)
target_sources(singleDemo
    PRIVATE
        "single_demo.cpp"
)

add_executable(Example1)
target_sources(Example1
    PRIVATE
        "example1/main.cpp"
        "example1/mytime.hpp"
)

add_executable(Example2)
target_sources(Example2
    PRIVATE
        "example2/main.cpp"
        "example2/mytime.hpp"
)

add_executable(Example3)
target_sources(Example3
    PRIVATE
        "example3/main.cpp"
        "example3/mytime.hpp"
)

add_executable(Example4)
target_sources(Example4
    PRIVATE
        "example4/main.cpp"
        "example4/mytime.hpp"
)

add_executable(Example5)
target_sources(Example5
    PRIVATE
        "example5/main.cpp"
        "example5/mytime.hpp"
)
