cmake_minimum_required(VERSION 3.25)

project("lwyu_example"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate using lwyu on an executable"
  LANGUAGES CXX
)

find_package(OpenSSL)

# only enable lwyu if openSSL is found and only for GCC and CLang
if(OpenSSL-NOTFOUND)
  message(STATUS "OpenSSL not found, disabling link what you use example")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  option(ENABLE_LINK_WHAT_YOU_USE "Enable link what you use" ON)

  add_executable(lwyu_example)
  target_sources(lwyu_example
    PRIVATE src/main.cpp
  )
  target_link_libraries(lwyu_example
    OpenSSL::SSL
  )
  set_target_properties(lwyu_example
    PROPERTIES LINK_WHAT_YOU_USE ${ENABLE_LINK_WHAT_YOU_USE}
  )
else()
  message(STATUS "Your compiler ${CMAKE_CXX_COMPILER_ID} does not support link what you use, disabling link what you use example")
endif()
