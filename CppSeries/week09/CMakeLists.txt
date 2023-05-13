cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(firstclass)
target_sources(firstclass
    PRIVATE
        "firstclass.cpp"
)

add_executable(constructor)
target_sources(constructor
    PRIVATE
        "constructor.cpp"
)

add_executable(destructor)
target_sources(destructor
    PRIVATE
        "destructor.cpp"
)

add_executable(const_class)
target_sources(const_class
    PRIVATE
        "const.cpp"
)

add_executable(static_class)
target_sources(static_class
    PRIVATE
        "static.cpp"
)

add_executable(this_pointer)
target_sources(this_pointer
    PRIVATE
        "this_pointer.cpp"
)

add_executable(log_class)
target_sources(log_class
    PRIVATE
        "log_class.cpp"
)
