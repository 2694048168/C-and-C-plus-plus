cmake_minimum_required(VERSION 3.25)

project("iwyu_example"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate using iwyu on an executable"
  LANGUAGES CXX
)

# if the iwyu program is found it is configured
find_program(INCLUDE_WHAT_YOU_USE_EXECUTABLE NAMES include-what-you-use iwyu)

if(INCLUDE_WHAT_YOU_USE_EXECUTABLE)
  option(ENABLE_INCLUDE_WHAT_YOU_USE "Enable iwyu" ON)

  list(APPEND INCLUDE_WHAT_YOU_USE_COMMAND "${INCLUDE_WHAT_YOU_USE_EXECUTABLE}"
    "-Xiwyu"
    "any"
    "-Xiwyu"
    "iwyu"
    "-Xiwyu"
    "-I ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}"
  )
endif()

add_executable(iwyu_example)
target_sources(iwyu_example
  PRIVATE src/main.cpp
)

if(ENABLE_INCLUDE_WHAT_YOU_USE)
  message(STATUS "Enabling IWYU for example ${INCLUDE_WHAT_YOU_USE_COMMAND}")
  set_target_properties(iwyu_example
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE_COMMAND}"
  )
endif()
