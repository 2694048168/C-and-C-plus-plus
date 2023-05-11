cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(variables)
target_sources(variables
    PRIVATE
        "variables.cpp"
)

add_executable(overflow)
target_sources(overflow
    PRIVATE
        "overflow.cpp"
)

add_executable(float)
target_sources(float
    PRIVATE
        "float.cpp"
)

add_executable(const_variable)
target_sources(const_variable
    PRIVATE
        "const_variable.cpp"
)

add_executable(DataInteger)
target_sources(DataInteger
    PRIVATE
        "exercises.cpp"
)
