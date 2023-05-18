## CMake tutorial

> the modern CMake tutorial for C++ build, examples about Executable Binary Programm, Static librarys and dynamic librarys, example about the Nest-style CMake and the graphviz releationship. We must to pay attention to the difference in loading dynamic libraries between Windows and Linux systems, that is, the symbol table import ways of dynamic libraries.

### Qucik Start
```shell
# git clone this repo. and enter the folder.
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/CMakeTutorial

# Setp 1. generate binary programm
cd ExeExample
cmake -B build # 平台默认编译工具链
cmake -B build -G Ninja
cmake --build build
./build/bin/main

# or as following
g++ myadd.cpp mysub.cpp mydiv.cpp mymul.cpp main.cpp -o main
clang++ myadd.cpp mysub.cpp mydiv.cpp mymul.cpp main.cpp -o main.exe

# Setp 2. generate static library
cd StaticLibExample
cmake -B build # 平台默认编译工具链 MSVC
cmake -B build -G "MinGW Makefiles" # or GCC
cmake -B build -G Ninja # or LLVM
cmake --build build
./build/bin/main

# or as following
g++ -c myadd.cpp mysub.cpp mydiv.cpp mymul.cpp
ar src libcalc.a myadd.o mysub.o mydiv.o mymul.o
g++ main.cpp -o main -L./ -lcalc

# Setp 3. generate dynamic library
cd DynamicLibExample
cmake -B build # 平台默认编译工具链
cmake -B build -G Ninja
cmake --build build
./build/bin/main

# or as following on Linux
# Linux 通过 -fPIC 导入符号表; Windows 通过 .lib 导入符号表
g++ -c myadd.cpp mysub.cpp mydiv.cpp mymul.cpp -fPIC
g++ -shared myadd.o mysub.o mydiv.o mymul.o -o libcalc.so
# g++ main.cpp -o main -L./ -lcalc -I./
g++ -Wl,-rpath=./ main.cpp -o main -L./ -lcalc -I./

# 动态链接器搜索路径
# 1. rpash 运行时路径(runtime path)
# 2. 系统变量 .bashrc
# 3. 系统目录 /usr/lib /usr/local/lib
# 4. ld.so.cache 动态库缓存中

# Setp 4. project with Nest CMake
cd NestCMakeExample
cmake -B build -G Ninja
cmake --build build
./bin/main

# Setp 5. the relationship from graphviz
cmake -B build --graphviz=build/visual_tree.dot -G Ninja
cmake --build build
cd build
dot -Tpng visual_tree.dot -o visual_tree.png
# seen in build/visual_tree.png

# Setp 6. the relationship from graphviz by custom_target_commmand
cd GraphvizCMake
cmake -B build
cmake --build build
# seen in build/visual_tree.png
```

### Organization of Repo.
```
. CMakeTutorial
|—— ExeExample
|   |—— mycalc.hpp
|   |—— mymul.cpp
|   |—— myadd.cpp
|   |—— mysub.cpp
|   |—— mydiv.cpp
|   |—— main.cpp
|   |—— CMakeLists.txt
|—— StaticLibExample
|   |—— calc
|   |—— |—— myadd.cpp
|   |—— |—— mysub.cpp
|   |—— |—— mymul.cpp
|   |—— |—— mydiv.cpp
|   |—— include
|   |—— |—— mycalc.hpp
|   |—— main.cpp
|   |—— CMakeLists.txt
|—— DynamicLibExample
|   |—— calc
|   |—— |—— myadd.cpp
|   |—— |—— mysub.cpp
|   |—— |—— mymul.cpp
|   |—— |—— mydiv.cpp
|   |—— |—— mycalc.hpp
|   |—— main.cpp
|   |—— CMakeLists.txt
|—— NestCMakeExample
|   |—— calc
|   |—— |—— myadd.cpp
|   |—— |—— mysub.cpp
|   |—— |—— mymul.cpp
|   |—— |—— mydiv.cpp
|   |—— |—— CMakeLists.txt
|   |—— include
|   |—— |—— mycalc.hpp
|   |—— src
|   |—— |—— main.cpp
|   |—— |—— CMakeLists.txt
|   |—— bin
|   |—— lib
|   |—— CMakeLists.txt
|—— GraphvizCMake
|   |—— CMakeLists.txt
|—— README.md
```
