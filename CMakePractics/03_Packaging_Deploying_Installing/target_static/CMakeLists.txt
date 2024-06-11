cmake_minimum_required(VERSION 3.25)

project(target_static
    VERSION 1.0.1
    DESCRIPTION "TARGET installable static library project"
    LANGUAGES CXX
)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME}
    PRIVATE src/hello.cpp
)
target_include_directories(${PROJECT_NAME}
    PUBLIC include
)
target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_17
)

# Defines the ${CMAKE_INSTALL_INCLUDEDIR} variable.
include(GNUInstallDirs)

# Make executable target `target_static` installable
install(TARGETS ${PROJECT_NAME})

# Install the header files.
# Since header files are not listed as output artifacts,
# they have to be installed separately.
install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
