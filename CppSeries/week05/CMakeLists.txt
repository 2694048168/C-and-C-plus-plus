cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(Pointers)
target_sources(Pointers
    PRIVATE
        "pointers.cpp"
)

add_executable(pointer_array)
target_sources(pointer_array
    PRIVATE
        "pointer_array.cpp"
)

add_executable(pointer_arithmetic)
target_sources(pointer_arithmetic
    PRIVATE
        "pointer_arithmetic.cpp"
)

add_executable(stack_heap)
target_sources(stack_heap
    PRIVATE
        "stack_heap.cpp"
)
