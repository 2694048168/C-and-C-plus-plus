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
|—— week02
|   |—— variables.cpp
|   |—— overflow.cpp
|   |—— float.cpp
|   |—— const_variable.cpp
|   |—— exercises.cpp
|   |—— CMakeLists.txt
|—— week03
|   |—— if_condition.cpp
|   |—— while_loop.cpp
|   |—— for_loop.cpp
|   |—— goto_statement.cpp
|   |—— switch_statement.cpp
|   |—— CMakeLists.txt
|—— week04
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
