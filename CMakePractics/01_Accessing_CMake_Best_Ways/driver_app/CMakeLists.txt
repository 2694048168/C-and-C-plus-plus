# Project directive is used for creating a child project for the targets we're
# going to create for the component. This will allow us to use project()
# variables in target creation.
project(
    _driver_application
    VERSION 1.0
    DESCRIPTION "Driver application implementation"
    LANGUAGES CXX)

add_executable(
    _driver_application) # Create an executable target named with the same name as

# project name
target_sources(_driver_application PRIVATE main.cpp) # Add main.cpp as

# source to the target
target_link_libraries(
    _driver_application PRIVATE _framework_component1 _framework_component2
) # Link target against ch2_framework_component1 and ch2_framework_component2

target_compile_features(_driver_application PUBLIC cxx_std_11)

install(
    TARGETS _driver_application
    COMPONENT executables
)
