#!/usr/bin/bash

rm -rf build
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build .