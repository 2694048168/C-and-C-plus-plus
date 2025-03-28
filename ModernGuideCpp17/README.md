## C++ Core Guidelines Explained and Modern C Plus Plus

> the Best Practices for Modern C++, the Compile & Linking & Loading & Library.

> Core Guidelines for Writing Clean, Safe and Fast Code

> What I cannot create, I do not understand. Know how to solve every problem that has been solved. Quote from Richard Feynman

> TODO list, we can quickly check in [VSCode](https://code.visualstudio.com/) with extension [Todo Tree](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree).

> What I cannot create, I do not understand. Know how to solve every problem that has been solved. Quote from Richard Feynman

**Overview**
- [feature](#features)
- [quick start](#quick-start)
- [smart pointer](#smart-pointer)
- [features in Modern C++](#features-in-modern-cpp)
- [File IO Write and Read via Modern C++](#file-io-write-and-read-via-modern-cpp)
- [compile linking](#compile-and-linking)
- [quick CMake](#quick-cmake)
- [compile and linking](#compile--linking--loading--library)

### **Features**
- [x] Modern C++ Tricks for Project Practice
- [x] Modern C++ Guide and Performance and Paradigm for Beginner
- [x] 异步学习之回调函数, std::vector 性能操作, std::map 注意事项
- [x] C++ 头文件后缀说明, **.h**表示只有声明的单纯头文件, **.hpp**表示声明和实现一起的头文件,如模板类的头文件等
- [x] 函数返回值类型后置 Trailing Return Type
- [x] Version of Application Software via CMake
- [x] Universally Unique Identifier for C++
- [x] The Modern C++ Thread Pool Asynchronous tasks
- [x] 发布-订阅（Publish-Subscribe）机制的C++实现
- [x] Precompiled Headers in C++ (Visual Studio setting and GCC setting)
- [x] Callable object and functional and bind in Modern C++
- [x] Windows Thread API into a ThreadBase class of Modern C++
- [x] Memory Leak check tool for C++ Valgrind(Linux platforms)
- [x] Global Variable in C++ Programming
- [x] MSVC & GCC & Clang compiler support
- [x] Modern C++ standard for C++11 & C++17
- [x] C++ CMake package management or VS build and VSCode
- [x] Build C++ phase include compile + linking + loading library
- [x] Smart pointer in Modern C++
- [x] Ownership and Move Semantic in Modern C++
- [x] Multiple threading in Modern C++
- [x] Callback function or methods in Modern C++
- [x] FileSystem library in Modern C++17
- [x] Virtual function and polymorphic in Modern C++
- [x] [Standard Predefined Macros](https://gcc.gnu.org/onlinedocs/cpp/Standard-Predefined-Macros.html) in C/C++
- [x] The C++ Core Guidelines(CG) presents its rules relatively tersely in a fixed format
- [x] The C++ Core Guidelines(CG) rules are often expressed in language-technical terms with an emphasis on enforcement through static analysis
- [x] The guidelines provide excellent, simple advice for improving your C++ style such that you can write correct, performant, and efficient code at your first attempt
- [x] [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [x] **Write in ISO Standard C++** and **Extension to C++**,使用ISO标准C++,而不使用C++编译器扩展

```shell
# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志(11,17,23), 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
```

- [x] **Use #include guards for all header files, NOT #pragma once**,使用header guards防止头文件包含,减少编译时间

```C++
// 现代编译器基本都支持该宏定义
#pragma once

// header guards
#ifndef __HEADER_GUARDS_H__
#define __HEADER_GUARDS_H__

void some_func(char str[]);

#endif /* __HEADER_GUARDS_H__ */

```

- [x] Modern C++11, new types were introduced in header <cstdint> that **defined fixed-width integral types**
- [x] **Backward compatibility Forward compatibility** and "Y2K"
- [x] [ISO CPP](https://isocpp.org/) and Cpp Conferences(CppCon)
- [x] **Prefer default arguments over overloading**, 抽象接口API时候优先使用默认参数而非重载
- [x] 不要定义仅初始化数据成员的默认构造函数而应使用类成员初始化
- [x] 避免平凡的get和set函数
- [x] 每条语句只声明一个名字
- [x] 不强求函数只用一条return语句
- [x] 将凌乱的结构封装起来，而不是使其散布于代码中
- [x] 尽量减少函数参数
- [x] 使用C风格子集获取跨编译器的ABI
- [x] 按成员声明顺序定义并初始化成员变量
- [x] 尽量减少可写数据的显式共享
- [x] 只在真正需要时使用模板元编程
- [x] 切勿通过原生指针（T*）或引用（T&）转移所有权
- [x] 避免使用单例
- [x] 依靠构造函数和赋值运算符，而不是 memset 和 memcpy
- [x] 不要用强制转换去除const限定符
- [x] 避免基于全局状态（如errno）的错误处理
- [x] 不要在头文件的全局作用域写using namespace
- [x] 优先选择结构体或元组返回多个“输出”值
- [x] 优先选择类枚举而不是“普通”枚举
- [x] 保持作用域小
- [x] 使用constexpr表示编译时可以计算的值
- [x] 使用模板提高代码的抽象层次
- [x] 为所有模板参数指定概念
- [x] 理想情况下，程序应具有静态类型安全性
- [x] 优先选择不可变数据而不是可变数据
- [x] 封装违反规则的部分
- [x] 确定初始值后再声明变量
- [x] 为促成优化而设计
- [x] 使用RAII防止泄露

### quick start

```shell
# git clone this repo. into local path
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/ModernGuideCpp17

# install the Clang(LLVM) and CMake and VSCode
code .

cmake or g++ or clang++
```

**the compilers for C++**
- [LLVM Clang download](https://releases.llvm.org/)
- [GCC download](https://gcc.gnu.org/releases.html)
- [MinGW download](https://winlibs.com/)
- [VS2022 cl download](https://visualstudio.microsoft.com/zh-hans/vs/)

```shell
gcc --version
# gcc.exe (MinGW-W64 x86_64-ucrt-posix-seh) 12.3.0
# gcc.exe (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders) 13.1.0

g++ --version
# g++.exe (MinGW-W64 x86_64-ucrt-posix-seh) 12.3.0
# g++.exe (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders) 13.1.0

clang --version
# clang version 16.0.0
# clang version 17.0.1

clang++
# clang version 16.0.0
# clang version 17.0.1

cl
# 用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.35.32217.1 版(VS2022)
```

> **you must to know the build process for C++, include the compile and link. More detail information at [here](https://2694048168.github.io/blog/#/PaperMD/cpp_env_test)**

```shell
g++ main.cpp -std=c++17 # main
g++ main.cpp -std=c++20
g++ main.cpp -std=c++23

clang++ main.cpp -std=c++17 # main
clang++ main.cpp -std=c++20
clang++ main.cpp -std=c++2b # for ISO C++23

cl main.cpp /std:c++17 /EHsc # main
```

### smart pointer

```shell
# 智能指针和 RAII 技术, 对象所需资源在其生命周期内始终保持有效, 同时不需要显式释放资源
mkdir smart_pointer && cd smart_pointer
touch main.cpp
touch CMakeLists.txt
mkdir utility
```

### features in Modern Cpp
- String literal, C++11 中添加了定义原始字符串的字面量
- NULL in C and nullptr in modern C++11
- const and constexpr in modern C++11 常量表达式和常量常量表达式函数
- auto keyword 自动类型推导
- final and override keyword
- C++11 Support 类型转换函数,数值类型和字符串类型之间的转换
- assert and static_assert in C++11
- noexcept 表示其修饰的函数不会抛出异常, or directly call 'std::terminate()' 函数来终止程序的运行
- chrono library in Modern C++11, 日期和时间相关的库
- thread library in Modern C++11, C++ 语言层面提供的线程库
- 多线程同步(Synchronous): mutex + condition_variable + atomic and atomic OP
- 多线程异步(Asynchronous): std::future + std::promise + std::packaged_task + std::async
- Special member function of a Class
- Friendship and inheritance for Class
- Polymorphism and (pure) virtual function in OOP
- Preprocessor directives in Modern C++ usefully


### File IO Write and Read via Modern Cpp
- write and read CSV file


### Compile and Linking

```shell
# C++ 的编译，链接的整个构建过程，Makefile or CMake
mkdir compile_linking && cd compile_linking
touch README.md
touch main.cpp

```

#### 编译工具
- GCC tool: gcc or g++
- Clang tool: clang or clang++
- Microsoft Visual C++ tool: cl

#### commands

```shell
touch main.cpp

# 1. compile phase
# gcc -c main.cpp -o main.o
g++ -c main.cpp -o main.o

# 2. linking phase
gcc main.o -o main -lstdc++
# gcc and g++ 的差异, g++ 不需要提供标准的C++库进行链接
g++ main.o -o main

# or compile and linking together
gcc main.cpp -o main -lstdc++
g++ main.cpp -o main

```

```shell
# c++ 编译过程: 预处理 --> 编译 --> 汇编 --> 链接
g++ -E main.cpp -o main.i
g++ -S main.i -o main.s
g++ -c main.s -o main.o
g++ main.o -o main

```

#### error

```shell
# so that error: compile error and linking error
touch add.cpp
touch add.hpp

g++ -c add.cpp -o add.o
# compile unit
g++ -c main.cpp -o main.o
g++ add.o main.o -o main

# or one command
g++ add.cpp main.cpp -o main

# why need header file,
# compile phase TYPE check, but not check type in linking
# the header file search path, 
# "add.h" 相对当前路径进行搜索; <add.h> 相对设置的头文件路径进行搜索, 默认有头文件搜索路径
# g++ add.cpp main.cpp -o main
g++ -I. add.cpp main.cpp -o main

```

#### multi-file

```shell
touch sub.hpp sub.cpp

# one command
# g++ sub.cpp add.cpp main.cpp -o main
g++ -I. sub.cpp add.cpp main.cpp -o main

# or compile and linking
g++ -I. -c sub.cpp -o sub.o
g++ -I. -c add.cpp -o add.o
g++ -I. -c main.cpp -o main.o
g++ sub.o add.o main.o -o main

```

#### CMake and Makefile
- C++ big project and many files to organize to build(compile and linking)
- save build time for big C++ project

```shell
# install make and cmake tool
make --version
mingw32-make --version

touch Makefile
# make target # default the first target
mingw32-make target
mingw32-make clean

```

```shell
# install cmake tool
touch CMakeLists.txt

# cmake build
cmake -S . -B build

cmake --build build --config Debug
./build/Debug/main

cmake --build build --config Release
./build/Release/main

# CMake 可以支持生成多种构建方式
cmake -G
rm -r build
cmake -S . -B build -G Ninja

```

```makefile
CFLAGS := -g -o2 -Wall -Werror -Wno-unused -ldl -std=c++17

target: sub_o add_o main_o main

main:
	g++ $(CFLAGS) sub.o add.o main.o -o main

sub_o:
	g++ $(CFLAGS) -I. -c sub.cpp -o sub.o
	
add_o:
	g++ $(CFLAGS) -I. -c add.cpp -o add.o

main_o:
	g++ $(CFLAGS) -I. -c main.cpp -o main.o

clean:
	rm -rf *.o main

```

```cmake
cmake_minimum_required(VERSION 3.20)

project(main
    VERSION "0.1.1"
    DESCRIPTION "the first CMake example"
    LANGUAGES CXX
)

# cmake --build build --config Release
set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")
endif()

message(STATUS "==== the project source dir: ${PROJECT_SOURCE_DIR}")
include_directories("./")

add_executable(main)
target_sources(main
    PRIVATE
        "add.cpp"
        "sub.cpp"
        "main.cpp"
)

```

### quick CMake

```shell
touch CMakeLists.txt
touch main.cpp

# cmake -S . -B build
cmake -S . -B build -G Ninja
# cmake -S . -B build -G "Unix Makefiles"
# cmake -S . -B build -G "Visual Studio 17 2022"
# cmake -S . -B build -G "Visual Studio 17 2019"
# cmake -S . -B build -G "MinGW Makefiles"
# cmake -S . -B build -G "MSYS Makefiles"

cmake --build build

./build/main
./build/test

```

```CMake
cmake_minimum_required(VERSION 3.20)

project(quick_cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER clang) # clang | gcc | MSVC(cl)
set(CMAKE_CXX_COMPILER g++) # clang++ | g++ | | MSVC(cl)

# -std=c11 std=c14 std=c17 std=c20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_EXTENSIONS OFF)

# -std=c99 std=c11 std=c18
# set(CMAKE_C_STANDARD 18)
# set(CMAKE_C_STANDARD_REQUIRED ON)
# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_C_EXTENSIONS OFF)

# 给后续的目标加上编译选项, 根据编译器而定, 这里设置的是 GCC compiler
add_compile_options(-g -Wunused)

# 添加头文件搜索路径
include_directories("./sub/" "./add/" "math/")

# 获取 src 目录下的所有.cpp 文件，并将其保存到变量中
file(GLOB_RECURSE SOURCES "sub/*.cpp" "add/*.cpp")

set(MUL_SOURCES "./math/mul.cpp")
# 指定从某些源文件创建库文件（静态库、动态库）
# add_library(mul STATIC ${MUL_SOURCES})
add_library(mul SHARED ${MUL_SOURCES})

# link_directories(./)
# link_libraries(mul)

add_executable(main ${SOURCES} main.cpp)
add_executable(test ${SOURCES} test.cpp)

# 给指定的目标加上编译选项
target_compile_options(main PUBLIC -Wall -Werror)

target_link_directories(main PUBLIC ./)
target_link_libraries(main mul)

# 根据功能模块构建编译, 指定到模块的 CMakeLists.txt
# add_subdirectory()

```

### Compile & Linking & Loading & Library

```C
#include <stdio.h>

int main(int argc, char *argv[])
{
    printf("hello world\n");
    return 0;
}
```

- 程序为什么要被编译器编译了之后才可以运行?
- 编译器在把C语言程序转换成可以执行的机器码的过程中做了什么，怎么做的?
- 最后编译出来的可执行文件里面是什么?除了机器码还有什么?它们怎么存放的，怎么组织的?
- #include <stdio.h>是什么意思?把stdio.h包含进来意味着什么?C语言库又是什么?它怎么实现的?
- 不同的编译器(MSVC, GCC or Clang)和不同的硬件平台(x86、SPARC、MIPS、ARM),以及不同的操作系统（Windows、Linux、UNIX、Solaris)，最终编译出来的结果一样吗?为什么?
- Hello World程序是怎么运行起来的?操作系统是怎么装载它的?它从哪儿开始执行，到哪儿结束? main函数之前发生了什么? main函数结束以后又发生了什么?
- 如果没有操作系统，Hello World可以运行吗?如果要在一台没有操作系统的机器上运行Hello World需要什么?应该怎么实现?
- printf 是怎么实现的?它为什么可以有不定数量的参数?为什么它能够在终端上输出字符串?
- Hello World 程序在运行时，它在内存中是什么样子的？


```C++
#include <iostream>

int main(int argc, const char **argv)
{
    std::cout << "hello Cpp world\n";

    return 0;
}
```

- 程序为什么要被编译器编译了之后才可以运行?
- 编译器在把C++语言程序转换成可以执行的机器码的过程中做了什么，怎么做的?
- 最后编译出来的可执行文件里面是什么?除了机器码还有什么?它们怎么存放的，怎么组织的?
- #include <iostream>是什么意思?把iostream头文件包含进来意味着什么?C++语言库又是什么?它怎么实现的?
- 不同的编译器(MSVC, GCC or Clang)和不同的硬件平台(x86、SPARC、MIPS、ARM),以及不同的操作系统（Windows、Linux、UNIX、Solaris)，最终编译出来的结果一样吗?为什么?
- Hello World程序是怎么运行起来的?操作系统是怎么装载它的?它从哪儿开始执行，到哪儿结束? main函数之前发生了什么? main函数结束以后又发生了什么?
- 如果没有操作系统，Hello World可以运行吗?如果要在一台没有操作系统的机器上运行Hello World需要什么?应该怎么实现?
- std::cout 是怎么实现的?数据流是什么?为什么它能够在终端上输出字符串?
- Hello World 程序在运行时，它在内存中是什么样子的？