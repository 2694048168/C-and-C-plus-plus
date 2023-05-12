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
