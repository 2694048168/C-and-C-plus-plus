cmake_minimum_required(VERSION 3.25)

project(library
  VERSION 1.0
  DESCRIPTION "CPack example"
  LANGUAGES CXX
)

add_library(${PROJECT_NAME}
  STATIC src/lib.cpp
)
target_compile_features(${PROJECT_NAME}
  PRIVATE cxx_std_17
)
target_include_directories(${PROJECT_NAME}
  PUBLIC include
)

# Defines the ${CMAKE_INSTALL_INCLUDEDIR} variable.
include(GNUInstallDirs)

# Make executable target `library` installable.
install(TARGETS ${PROJECT_NAME})

# Install the header files. 
# Since header files are not listed as output artifacts, they have
# to be installed separately.
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
