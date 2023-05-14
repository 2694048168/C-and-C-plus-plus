cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(friendClass)
target_sources(friendClass
    PRIVATE
        "friend_class.cpp"
)

add_executable(friendClassDemo)
target_sources(friendClassDemo
    PRIVATE
        "friend_problem.cpp"
)

add_executable(NestedEnum1)
target_sources(NestedEnum1
    PRIVATE
        "nested_enum1.cpp"
)

add_executable(NestedEnum2)
target_sources(NestedEnum2
    PRIVATE
        "nested_enum2.cpp"
)

add_executable(NestedClass)
target_sources(NestedClass
    PRIVATE
        "nested_class.cpp"
)

add_executable(RTTI)
target_sources(RTTI
    PRIVATE
        "rtti.cpp"
)

add_executable(type_conversion)
target_sources(type_conversion
    PRIVATE
        "type_conversion.cpp"
)
