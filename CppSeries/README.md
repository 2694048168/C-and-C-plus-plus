## CPP tutorial with CMake and Clang++

> the modern C++ tutorial with [CMake](https://cmake.org/) and [Ninja](https://ninja-build.org/) build-tool and [Clang++](https://releases.llvm.org/download.html) compiler in [VSCode](https://code.visualstudio.com/) from [Shiqi Yu Prof.](https://github.com/ShiqiYu/CPP) and the [tutorial video](https://www.bilibili.com/video/BV1Vf4y1P7pq/) on Bilibili

### Qucik Start
```shell
# git clone this repo. and enter the folder.
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/CppSeries

# cmake build(compile and link)
cmake -B build -G Ninja
cmake --build build

# or g++/clang++ for single source cpp file
g++ hello.cpp -std=c++17 -o main
clang++ hello.cpp -std=c++17 -o main

# in the 'CppSeries/week10/opencv_mat' folder,
# you should build(compile and link) individually with CMake,
# the more detail information seen in 'README.md' file in this folder.
```


### Organization of Repo.
```
. Project_Name
|—— week01
|   |—— arithmetic
|   |—— |—— mymul.hpp
|   |—— |—— mymul.cpp
|   |—— hello.cpp
|   |—— main.cpp
|   |—— exercises.cpp
|   |—— CMakeLists.txt
|—— week04
|   |—— array.cpp
|   |—— stdstring.cpp
|   |—— struct.cpp
|   |—— union.cpp
|   |—— enum.cpp
|   |—— src
|   |—— |—— main.cpp
|   |—— |—— factorial.cpp
|   |—— |—— printhello.cpp
|   |—— |—— function.hpp
|   |—— CMakeLists.txt
|—— week05
|   |—— pointers.cpp
|   |—— pointer_array.cpp
|   |—— pointer_arithmetic.cpp
|   |—— stack_heap.cpp
|   |—— CMakeLists.txt
|—— week06
|   |—— src
|   |—— |—— main.cpp
|   |—— |—— math
|   |—— |—— |—— mymath.hpp
|   |—— |—— |—— mymath.cpp
|   |—— basic_function.cpp
|   |—— inline_function.cpp
|   |—— param_pointer.cpp
|   |—— param_reference.cpp
|   |—— CMakeLists.txt
|—— week08
|   |—— main.cpp
|   |—— matoperation.cpp
|   |—— matoperation.hpp
|   |—— CMakeLists.txt
|—— build
|—— bin
|—— CMakeLists.txt
|—— README.md
```

### Context of Repo.
- week01: Getting Started
- week02: Data Types and Arithmetic Operators
- week03: Branching and Looping Statements
- week04: Data Structures
- week05: Pointers and Dynamic Memory Management
- week06: Basics of Functions
- week07: Advances in Functions
- week08: Speedup Your Program
- week09: Basics of Classes
- week10: Advances in Classes
- week11: Dynamic Memory Management in Classes
- week12: Class Inheritance and virtual function(polymorphic)
- week13: Class Templates and std Library
- week14: Error Handling
- week15: Nested Classes and RTTI
