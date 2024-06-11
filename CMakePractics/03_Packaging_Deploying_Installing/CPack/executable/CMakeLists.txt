cmake_minimum_required(VERSION 3.25)

project(executable
  VERSION 1.0
  DESCRIPTION "CPack example"
  LANGUAGES CXX
)

add_executable(${PROJECT_NAME}
  src/main.cpp
)
target_compile_features(${PROJECT_NAME}
  PRIVATE cxx_std_17
)

# Link executable target to library target
target_link_libraries(${PROJECT_NAME}
  PRIVATE library
)

install(TARGETS ${PROJECT_NAME})
