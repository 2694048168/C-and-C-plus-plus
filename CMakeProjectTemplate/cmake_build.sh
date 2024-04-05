#!/bin/bash

# cmake -S . -B build
cmake -S . -B build -G Ninja

# Build debug binaries
# cmake --build build --config Debug

# Build release binaries
cmake --build build --config Release
