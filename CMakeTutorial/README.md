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

cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Debug
cmake -S . -B build -DCMAKE_BUILD_TYPE:STRING=Debug
cmake --build build --config Debug

cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Release
cmake -S . -B build -DCMAKE_BUILD_TYPE:STRING=Release
cmake --build build --config Release

cmake --build build --target clean

# 自动以最大线程数进行并行编译
sudo cmake --build build --target all -j12
```

### CMake snippets
```shell
if(WIN32)
    # Windows 上使用特定的库
    set(MY_LIBRARY "C:/libs/windows_lib.lib")
    # 打印调试信息
    message("Using library for Windows")
elseif(APPLE)
    # macOS 上使用其他库
    set(MY_LIBRARY "/usr/local/lib/mac_lib.dylib")
    message("Using library for macOS")
else()
    # Linux 上用另一个库
    set(MY_LIBRARY "/usr/lib/linux_lib.so")
    message("Using library for Linux")
endif()

# 最终链接库
target_link_libraries(my_project ${MY_LIBRARY})

# CMake 高效配置：条件判断、循环与字符串操作
# 循环：foreach 和 while
set(SOURCES main.cpp utils.cpp helper.cpp)

foreach(SRC ${SOURCES})
    message("Compiling source file: ${SRC}")
    # 这里可以做更多操作，比如添加每个源文件到目标中
    target_sources(hello PRIVATE ${SRC})
endforeach()

set(COUNT 1)

while(COUNT LESS 5)
    message("Current count: ${COUNT}")
    math(EXPR COUNT "${COUNT} + 1")
endwhile()

# 字符串操作：拼接、分割、查找
set(DIR "/home/user/project/")
set(FILE_NAME "main.cpp")

# 拼接字符串
set(FULL_PATH "${DIR}${FILE_NAME}")

message("Full path: ${FULL_PATH}")

# 字符串分割（STRING(SUBSTRING ...)）
set(PATH "/home/user/project/main.cpp")

# 获取文件名
string(REGEX REPLACE "^.*/" "" FILE_NAME ${PATH})
# ^ 是一个锚点，表示匹配字符串的开始位置。 
# .* 会匹配任意数量的字符，甚至是零个字符。它会一直匹配，直到遇到我们指定的下一个字符(/)。

message("File name: ${FILE_NAME}")

# 字符串查找（STRING(FIND ...)）
set(PATH "/home/user/project/")

# 查找字符串中的子串
string(FIND ${PATH} "user" POSITION)

# 如果找到了，POSITION 会返回位置，找不到则返回 -1
if(POSITION GREATER -1)
    message("Found 'user' in the path at position: ${POSITION}")
else()
    message("'user' not found in the path.")
endif()

# Step2 ============= 使用 vcpkg tool 来管理依赖 =============
# 设置 vcpkg 的工具链文件
set(CMAKE_TOOLCHAIN_FILE "path_to_vcpkg/scripts/buildsystems/vcpkg.cmake")

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 链接到你的目标
target_link_libraries(my_target PRIVATE ${OpenCV_LIBS})

# Step2 ============= 使用 conan tool 来管理依赖 =============
# 设置 Conan 的工具链文件：加载由 Conan 生成的 conanbuildinfo.cmake 文件。
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# 应用加载的 conanbuildinfo.cmake 文件中的配置信息，设置项目的依赖环境。
conan_basic_setup()

# 使用 Boost 库
target_link_libraries(my_target PRIVATE Boost::Boost)

# Step3 ============= 使用 Git 子模块来管理依赖 =============
# 引入 Git 子模块
add_subdirectory(external/boost)

# 使用 boost 库
target_link_libraries(my_target PRIVATE boost)

```

### CMake中的 Release、Debug、MinSizeRel、RelWithDebInfo
- CMake中编译类型(build type)是指一组预定义的编译器和链接器标志(flag),用于控制生成的可执行文件或库的特性
- 常见的编译类型包括: Release、Debug、MinSizeRel、RelWithDebInfo

> Debug 编译类型包含了**详细的调试信息**, 如符号信息(变量名、函数名等)和行号信息等, 以便于调试器(如gdb,lldb,msvc)进行单步调试、变量监视从而可以准确地定位问题; 通常该类型的可执行文件或库主要是用在**开发和调试**阶段,方便程序员定位和修复错误, Debug 编译类型通常不对代码进行优化, 以避免优化代码的**执行顺序或逻辑**(这会造成bug位置不清晰而难以追溯), 因此**执行速度较慢**, 且因为增加了额外的调试信息而**体积变大**. Debug 编译类型用在需要详细调试程序、检查逻辑错误或分析运行时行为的场景, 其典型的编译器标志如下

```shell
# GCC/Clang: -g（生成调试信息）、-O0（禁用优化）
# MSVC: /zi（生成完整的调试信息）、/Od（禁用优化）
```

> Release 编译类型主要用于发布给最终用户,追求代码大小和**运行速度的最优**; **编译器**对代码进行**各种优化**以提高执行效率, 并且不包含调试信息, 从而减小可执行文件大小且保护源码实现细节. 因不包含调试信息, 所以不利于代码调试, 适用于代码经过反复调试和验证后准备上线发布时使用. Release 编译类型的可执行文件体积小、执行速度快,用于程序发布、性能测试或需要高效率运行的场合, 其典型的编译器标志如下

```shell
# GCC/Clang: -O3或-O2（高级别优化）、-DNDEBUG（禁用assert等调试宏）
# MSVC: /O2（全局优化）、/DNDEBUG（禁用assert等调试宏）
```

> MinSizeRel 是Minimum Size Release, 即最小尺寸发布; 见名知意,它的目的就是用来生成尺寸最小的可执行文件或库, 适用于存储空间有限的场景(比如嵌入式环境); 该类型的目的是在保证一定性能的前提下, 最大程度减小生成文件的体积, 既然其目的是尽可能使体积减小, 因此同Release一样, 它也不包含调试信息, 但不同于Release, 它的终极目的还是追求**体积的极致**, 因此可能会**牺牲一定的性能**. MinSizeRel 编译类型通常用于嵌入式设备、移动应用或对存储空间敏感的项目中, 其典型的编译器标志如下

```shell
# GCC/Clang: -Os（优化尺寸）、-DNDEBUG（禁用assert等调试宏）
# MSVC: /O1（最小化空间）
```

> RelWithDebInfo 是Release With Debug Info的缩写, 它是带调试信息的发布版, 兼顾了性能优化和调试能力,用于需要**在优化后的代码中进行调试**的情况. 该类型具备Release的特点, 启用了带阿米优化, 但又保留了一定的调试信息, 以便在出现问题时可以进行调试. 程序大小和性能均介于Debug和Release之间, 适合在需要优化性能的同时, 也需要调试信息的场景. RelWithDebInfo 编译类型的调试信息不如Debug丰富, 但比Release要丰富; 当需要调试仅在优化后出现的问题, 如**内存泄漏、线程竞争等，以及性能测试**时, 需要获取性能分析的数据的场景, 其典型的编译器标志如下

```shell
# GCC/Clang: -O2（优化）、-g（生成调试信息）、-DNDEBUG（禁用assert等调试宏）
# MSVC: /zi（调试信息）、/O2（优化）
```

#### **4种编译类型从不同维度进行总结并对比**

|           |           |      |
|  ----     | ----      |----  |
| 调试信息 Debug Info | 包含调试信息 | Debug、RelWithDebInfo |
| 调试信息 Debug Info | 不包含调试信息 | Release、MinSizeRel |
| 优化级别 Optimization Level | 禁用优化 | Debug |
| 优化级别 Optimization Level | 启用优化 | 高级别性能优化: Release、RelWithDebInfo |
| 优化级别 Optimization Level | 启用优化 | 尺寸优化: MinSizeRel |
| 文件大小 Output Size | 较大 | Debug: 包含调试符号且未优化 |
| 文件大小 Output Size | 较小 | Release、RelWithDebInfo: 优化后代码更紧凑 |
| 文件大小 Output Size | 最小 | MinSizeRel: 专门优化尺寸 |

**如何知道我们的项目所使用的不同编译选项的默认值是什么呢？**
```shell
# 在终端执行以下命令
cmake -LAH "CMakeLists.txt文件所在路径" | grep -C1 CMAKE_CXX_FLAGS

# 主CMakeLists.txt文件的末尾去打印想要了解的编译类型的默认编译器标志
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Library Type: " ${LIB_TYPE})
MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
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
