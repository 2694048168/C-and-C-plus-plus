![CPlusPlus Logo](./Logo.png)

> C/C++ language learning and some examples; OpenCV with C++; CMake with C++; CUDA with C++; OpenGL with C++; Vulkan with Modern C++; the Modern C++ guide and features and Multi Threading in Modern C++; the QT and GUI with C++. 

> [Wei Li Blog](https://2694048168.github.io/blog/)

**features**
- [x] Modern C++
- [x] Support CMake
- [x] Support GCC/Clang/MSVC
- [x] Support VSCode
- [x] Support CUDA
- [x] Support OpenGL
- [x] Support OpenCV
- [ ] Support Vulkan
- [x] Support QT with OpenCV and OpenGL


**overview**
- [quick start](#quick-start)
- [C/C++ Language Learning](#cc-language-learning)
- [Organization of Repo.](#organization-of-repo)
- [CUDA with C++](#cuda-with-c)
- [OpenGL with C++](#opengl-with-c)
- [CMake with C++](#cmake-with-c)
- [OpenCV with C++](#openccv-with-c)


### Quick Start

```shell
# step 1: clone the repo. and into the folder 'CppImageProcessing'
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git

# 查看每一个文件夹下的 'README.md' 文件说明
# if you want to download specical folder, please using 'gitzip' tool to enter
# the folder path, such as 'https://github.com/2694048168/C-and-C-plus-plus/tree/master/CppImageProcessing'.
# [gitzip](http://kinolien.github.io/gitzip/)
```

### Organization of Repo.
```
. C-and-C-plus-plus
|—— MultithreadingModernCpp
|   |—— README.md
|—— CMakeClangVcpkg
|   |—— vcpkg.json
|   |—— CMakePreset.json
|   |—— CMakeLists.txt
|   |—— .vscode
|   |—— |—— launch.json
|   |—— |—— tasks.json
|   |—— README.md
|—— CMakeTutorial
|   |—— ExeExample
|   |—— StaticLibExample
|   |—— DynamicLibExample
|   |—— NestCMakeExample
|   |—— GraphvizCMake
|   |—— README.md
|—— ModernGuideCpp17
|   |—— README.md
|—— CppSeries
|   |—— week01
|   |—— |—— CMakeLists.txt
|   |—— week02
|   |—— |—— CMakeLists.txt
|   |—— CMakeLists.txt
|   |—— README.md
|—— CppImageProcessing
|   |—— cuda_opencv
|   |—— images
|   |—— CMakeLists.txt
|   |—— vcpkg.json
|   |—— README.md
|—— CUDA_Programming
|   |—— 01_kernelFunction
|   |—— CMakeLists.txt
|   |—— README.md
|—— OpenGL_Cherno
|   |—— 00_opengl
|   |—— external
|   |—— |—— GLFW
|   |—— |—— GLEW
|   |—— rendering_pipeline.png
|   |—— CMakeLists.txt
|   |—— README.md
|—— OpenGL_Learning
|   |—— 00_hello_opengl
|   |—— resources
|   |—— thirdparty
|   |—— |—— GLFW
|   |—— |—— GLDA
|   |—— |—— GLM
|   |—— |—— STB
|   |—— CMakeLists.txt
|   |—— README.md
|—— CMakeReadme.md
|—— cmake_workflow.png
|—— Logo.png
|—— README.md
```

### CUDA with C++
- CUDA_CPlusPlus
- CUDA_Programming
<details>
<summary> <span style="color:PeachPuff">the CUDA Heterogeneous Programming with C++ via CMake.</span> </summary>


**the organization of project**
```
. CUDA_Programming
|—— 01_kernelFunction
|   |—— main.cu
|   |—— CMakeLists.txt
|—— 05_CudaError
|   |—— include
|   |—— |—— add.cuh
|   |—— |—— cudaError.cuh
|   |—— src
|   |—— |—— add.cu
|   |—— |—— cudaError.cu
|   |—— CMakeLists.txt
|—— CMakeLists.txt
|—— README.md
```

</details>

### OpenGL with C++
- OpenGL_Cherno
- OpenGL_Learning
<details>
<summary> <span style="color:PeachPuff">Computer graphics is the art or science of producing graphical images with the aid of computer.</span> </summary>


**the basic development environment**
```shell
# Modern code editor: Visual Studio Code
winget show code

# Build tool: CMake
cmake --version
# cmake version 3.26.3

# Compile tool: g++ from MinGW-64
g++ --version
gcc --version
mingw32-make --version
# g++.exe (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders) 12.2.0
# gcc.exe (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders) 12.2.0
# GNU Make 4.4 Built for x86_64-w64-mingw32

# GLFW: Windows pre-compiled binaries || 64-bit Windows binaries

# GLAD: Language(C/C++) || Specification(OpenGL) || Profile(Core) 
# || API-gl(Version 4.6) || others(None) ---> generate
```

**the organization of project**
```
. LearningOpenGL
|—— 00_hello_opengl
|   |—— src
|   |—— |—— hello_opengl.cpp
|   |—— CMakeLists.txt
|—— 01_hello_triangle
|   |—— src
|   |—— |—— hello_triangle.cpp
|   |—— CMakeLists.txt
|—— 02_rectangular
|—— 03_shader
|—— 03_shader
|—— 04_uniform
|—— 05_attribute
|—— 06_texture
|—— 07_transformations
|—— 08_coordinate
|—— 09_camera
|   |—— include
|   |—— |—— utils.hpp
|   |—— |—— shader.hpp
|   |—— shader
|   |—— |—— fragment.glsl
|   |—— |—— vertex.glsl
|   |—— src
|   |—— |—— camera_keyboard.cpp
|   |—— |—— camera_mouse_zoom.cpp
|   |—— |—— camera.cpp
|   |—— |—— shader.cpp
|   |—— |—— utils.cpp
|   |—— CMakeLists.txt
|—— thirdparty
|   |—— GLAD
|   |—— |—— include
|   |—— |—— src
|   |—— GLFW
|   |—— |—— include
|   |—— |—— lib
|   |—— STB
|   |—— |—— stb_image.h
|   |—— GLM
|   |—— |—— cmake
|   |—— |—— glm
|   |—— |—— |—— common.hpp
|—— bin
|—— lib
|—— build
|—— CMakeLists.txt
|—— images
|   |—— rendering_pileline.png
|—— README.md
```

**some Useful linker**
- [GLFW](https://www.glfw.org/download.html)
- [GLAD](https://glad.dav1d.de/)
- [STB](https://github.com/nothings/stb)
- [GLM](https://github.com/g-truc/glm/releases)
- [Learning OpenGL](https://learnopengl.com/)
- [Learning OpenGL 中文](https://learnopengl-cn.github.io/)
- [OpenGL Extension Viewer tool](https://download.cnet.com/OpenGL-Extensions-Viewer/3000-18487_4-34442.html)

**Basic Concepts**
- Graphics and Image
- Computer Graphics & Digital Image Processing & Computer Vision
- OpenGL identifier rule: <库前缀><根命令><可选参数数量><可选参数类型> "glColor3f"
- OpenGL Graphics Pipeline: the process of transforming 3D coordinates to 2D pixels
- Traditional rendering pipeline VS Volume rendering algorithm
- Shader and OpenGL Shading Language(GLSL)
- Primitive and Primitive Assembly
- Normalized Device Coordinates,NDC

> 在OpenGL中, 任何事物都在3D空间中, 而屏幕和窗口却是2D像素数组, 这导致OpenGL的大部分工作都是关于把3D坐标转变为适应你屏幕的2D像素; 3D坐标转为2D坐标的处理过程是由OpenGL的图形渲染管线(Graphics Pipeline, 大多译为管线, 实际上指的是一堆原始图形数据途经一个输送管道, 期间经过各种变化处理最终出现在屏幕的过程)管理的; 图形渲染管线可以被划分为两个主要部分: 第一部分把你的3D坐标转换为2D坐标; 第二部分是把2D坐标转变为实际的有颜色的像素; 2D坐标和像素也是不同的, 2D坐标精确表示一个点在2D空间中的位置, 而2D像素是这个点的近似值, 2D像素受到你的屏幕/窗口分辨率的限制; 图形渲染管线可以被划分为几个阶段, 每个阶段将会把前一个阶段的输出作为输入; 所有这些阶段都是高度专门化的(它们都有一个特定的函数), 并且很容易并行执行, 正是由于它们具有并行执行的特性, 当今大多数显卡都有成千上万的小处理核心, 它们在GPU上为每一个渲染管线阶段运行各自的小程序, 从而在图形渲染管线中快速处理你的数据, 这些小程序叫做着色器(Shader). 有些着色器可以由开发者配置, 因为允许用自己写的着色器来代替默认的, 所以能够更细致地控制图形渲染管线中的特定部分了, 因为它们运行在GPU上, 所以节省了宝贵的CPU时间, OpenGL着色器是用OpenGL着色器语言(OpenGL Shading Language, GLSL)写成的.

> Rendering Pipeline, 物体的顶点在最终转化为屏幕坐标之前还会被变换到多个坐标系统(Coordinate System); 将物体的坐标变换到几个过渡坐标系(Intermediate Coordinate System)的优点在于, 在这些特定的坐标系统中, 一些操作或运算更加方便和容易.

</details>

### CMake with C++
- CMakeClangVcpkg
<details>
<summary> <span style="color:PeachPuff">the modern for C++ with the modern toolchains, include CMake, vcpkg, Ninja, Clang and Git in VSCode.</span> </summary>

</details>

- CMakeTutorial
<details>
<summary> <span style="color:PeachPuff">the modern CMake tutorial for C++ build, examples about Executable Binary Programm, Static librarys and dynamic librarys, example about the Nest-style CMake and the graphviz releationship. We must to pay attention to the difference in loading dynamic libraries between Windows and Linux systems, that is, the symbol table import ways of dynamic libraries.</span> </summary>

</details>

- CMakeExamples
- CMakeReadme.md

### OpencCV with C++
- CppImageProcessing
<details>
<summary> <span style="color:PeachPuff">Image Processing Opertors and Algorithms with c++ via OpenCV libarary.</span> </summary>

**Quick Start**
```shell
# step 1: clone the repo. and into the folder 'CppImageProcessing'
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/CppImageProcessing

# install C++ package manager 'vcpkg'
# step 2: modify the 'vspkg' install path in the top 'CMakeLists.txt' file.
set(CMAKE_TOOLCHAIN_FILE "[path to vcpkg]/scripts/buildsystems/vcpkg.cmake")

# it may be time consumming in the first time build,
# because of downloading and building the OpenCV library.
# CMake configuration and build(compiler and linker)
cmake -B build
cmake --build build

# enter into 'bin' and running the demo example, such as 'RandomText'
cd bin
./RandomText
./CudaOpenCV
```

**Useful Links**
- [OpenCV](https://github.com/opencv/opencv)
- [OpenCV contrib](https://github.com/opencv/opencv_contrib)
- [OpenCV imgproc module](https://docs.opencv.org/4.7.0/d7/da8/tutorial_table_of_content_imgproc.html)
- [vcpkg](https://vcpkg.io/en/getting-started.html)
- [CMake](https://cmake.org/download/)
- [Git](https://git-scm.com/downloads)

**Organization of Project**
```
. CppImageProcessing
|—— hello_start
|   |—— main.cpp
|   |—— CMakeLists.txt
|—— build
|   |—— |—— vcpkg_installed
|   |—— |—— |—— x64-windows
|   |—— |—— |—— |—— bin
|   |—— |—— |—— |—— lib
|   |—— |—— |—— |—— include
|—— CMakeLists.txt
|—— vcpkg.json
|—— bin
|—— lib
|—— images
|—— README.md
```

</details>

- OpenCV_Linux_Ubuntu
- OpenCV-CPP

### C/C++ Language Learning
- C++PrimerPlus6thExercise
<details>
<summary> <span style="color:PeachPuff">C++ Primer Plus 第6版中文版 编程练习答案</span> </summary>

</details>

- CppConcurrencyAction
<details>
<summary> <span style="color:PeachPuff">现代 C++ 多线程编程和并发编程实战</span> </summary>

</details>

- MultithreadingModernCpp
<details>
<summary> <span style="color:PeachPuff">the multi-threading in Modern C++, including the thread | mutex | lock | conditional variable | atomic operator, etc.</span> </summary>

</details>

- ModernGuideCpp17
<details>
<summary> <span style="color:PeachPuff">the Modern C++ features and use examples for C++11, C++17 and C++23.</span> </summary>

</details>

- CppSeries
<details>
<summary> <span style="color:PeachPuff">the modern C++ tutorial with CMake and Ninja build-tool and Clang++ compiler in VSCode from Shiqi Yu Prof. and the tutorial video on Bilibili</span> </summary>

**CPP tutorial with CMake and Clang++**
> the modern C++ tutorial with [CMake](https://cmake.org/) and [Ninja](https://ninja-build.org/) build-tool and [Clang++](https://releases.llvm.org/download.html) compiler in [VSCode](https://code.visualstudio.com/) from [Shiqi Yu Prof.](https://github.com/ShiqiYu/CPP) and the [tutorial video](https://www.bilibili.com/video/BV1Vf4y1P7pq/) on Bilibili

**Qucik Start**
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

**Organization of Repo.**
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

**Context of Repo.**
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

</details>

- 21days_CPlusPlus
- Address-Book-System
- EffectiveCPlusPlus
- Essential-C++
- mnist_MachineLearning
- Modern_CPP_Overview
- PrimerC-plus-plus_exercises
- Staff-Management-System
- Student-Management-System
- Tetris-game
