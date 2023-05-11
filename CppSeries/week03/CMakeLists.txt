cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(if_condition)
target_sources(if_condition
    PRIVATE
        "if_condition.cpp"
)

add_executable(while_loop)
target_sources(while_loop
    PRIVATE
        "while_loop.cpp"
)

add_executable(for_loop)
target_sources(for_loop
    PRIVATE
        "for_loop.cpp"
)

add_executable(goto_statement)
target_sources(goto_statement
    PRIVATE
        "goto_statement.cpp"
)

add_executable(switch_statement)
target_sources(switch_statement
    PRIVATE
        "switch_statement.cpp"
)
