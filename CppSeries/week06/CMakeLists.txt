cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(basic_function)
target_sources(basic_function
    PRIVATE
        "basic_function.cpp"
)

add_executable(inline_function)
target_sources(inline_function
    PRIVATE
        "inline_function.cpp"
)

add_executable(param_pointer)
target_sources(param_pointer
    PRIVATE
        "param_pointer.cpp"
)

add_executable(param_reference)
target_sources(param_reference
    PRIVATE
        "param_reference.cpp"
)

add_executable(lab_math)
target_sources(lab_math
    PRIVATE
        "src/main.cpp"
        "src/math/mymath.cpp"
        "src/math/mymath.hpp"
)
