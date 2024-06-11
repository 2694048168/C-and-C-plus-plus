cmake_minimum_required(VERSION 3.25)

project(
    executable_target
    VERSION 1.0
    DESCRIPTION "TARGET installable executable project"
    LANGUAGES CXX)

add_executable(target_executable)
target_sources(target_executable
    PRIVATE main.cpp
)
target_compile_features(target_executable
    PRIVATE cxx_std_17
)

# Make executable target `target_executable` installable.
# this will only install the output artifacts produced by the target.
install(TARGETS target_executable)

# To override the built-in defaults,
install(TARGETS target_executable
    RUNTIME DESTINATION custom_bin
)

# cmake -S . -B build
# ====for single-config
# cmake --build build
# ====for multi-config
# cmake --build build --config Release
# cmake --install ./build --prefix ./bin
