cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(default_argument)
target_sources(default_argument
    PRIVATE
        "default_argument.cpp"
)

add_executable(overload)
target_sources(overload
    PRIVATE
        "overload.cpp"
)

add_executable(template_function)
target_sources(template_function
    PRIVATE
        "template_function.cpp"
)

add_executable(specialization)
target_sources(specialization
    PRIVATE
        "specialization.cpp"
)

add_executable(recursion)
target_sources(recursion
    PRIVATE
        "recursion.cpp"
)

add_executable(function_pointer)
target_sources(function_pointer
    PRIVATE
        "function_pointer.cpp"
)
