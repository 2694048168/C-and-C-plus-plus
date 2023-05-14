cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(stderror)
target_sources(stderror
    PRIVATE
        "stderror.cpp"
)

add_executable(assert)
target_sources(assert
    PRIVATE
        "assert.cpp"
)
target_compile_definitions(assert
  PRIVATE
    "-DNDEBUG"
)

add_executable(throw)
target_sources(throw
    PRIVATE
        "throw.cpp"
)

add_executable(throw_wrapper)
target_sources(throw_wrapper
    PRIVATE
        "throw_wrapper.cpp"
)
