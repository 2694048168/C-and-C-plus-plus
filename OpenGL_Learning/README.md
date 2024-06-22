## Fundamentals of Computer Graphics

> Computer graphics is the art or science of producing graphical images with the aid of computer. The reference book is: <陆枫 计算机图形学基础(第三版) 电子工业出版社> and <Steve&Peter Fundamentals of Computer Graphics(fifth edition)>

![](images/output.gif)

### Overview
- [the basic development environment](#the-basic-development-environment)
- [the organization of project](#the-organization-of-project)
- [some Useful linker](#some-useful-linker)
- [Basic Concepts](#basic-concepts)


#### **the basic development environment**
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

#### **the organization of project**
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

#### **some Useful linker**
- [GLFW](https://www.glfw.org/download.html)
- [GLAD](https://glad.dav1d.de/)
- [STB](https://github.com/nothings/stb)
- [GLM](https://github.com/g-truc/glm/releases)
- [Learning OpenGL](https://learnopengl.com/)
- [Learning OpenGL 中文](https://learnopengl-cn.github.io/)
- [OpenGL Extension Viewer tool](https://download.cnet.com/OpenGL-Extensions-Viewer/3000-18487_4-34442.html)

#### **Basic Concepts**
- Graphics and Image
- Computer Graphics & Digital Image Processing & Computer Vision
- OpenGL identifier rule: <库前缀><根命令><可选参数数量><可选参数类型> "glColor3f"
- OpenGL Graphics Pipeline: the process of transforming 3D coordinates to 2D pixels
- Traditional rendering pipeline VS Volume rendering algorithm
- Shader and OpenGL Shading Language(GLSL)
- Primitive and Primitive Assembly
- Normalized Device Coordinates,NDC

> 在OpenGL中, 任何事物都在3D空间中, 而屏幕和窗口却是2D像素数组, 这导致OpenGL的大部分工作都是关于把3D坐标转变为适应你屏幕的2D像素; 3D坐标转为2D坐标的处理过程是由OpenGL的图形渲染管线(Graphics Pipeline, 大多译为管线, 实际上指的是一堆原始图形数据途经一个输送管道, 期间经过各种变化处理最终出现在屏幕的过程)管理的; 图形渲染管线可以被划分为两个主要部分: 第一部分把你的3D坐标转换为2D坐标; 第二部分是把2D坐标转变为实际的有颜色的像素; 2D坐标和像素也是不同的, 2D坐标精确表示一个点在2D空间中的位置, 而2D像素是这个点的近似值, 2D像素受到你的屏幕/窗口分辨率的限制; 图形渲染管线可以被划分为几个阶段, 每个阶段将会把前一个阶段的输出作为输入; 所有这些阶段都是高度专门化的(它们都有一个特定的函数), 并且很容易并行执行, 正是由于它们具有并行执行的特性, 当今大多数显卡都有成千上万的小处理核心, 它们在GPU上为每一个渲染管线阶段运行各自的小程序, 从而在图形渲染管线中快速处理你的数据, 这些小程序叫做着色器(Shader). 有些着色器可以由开发者配置, 因为允许用自己写的着色器来代替默认的, 所以能够更细致地控制图形渲染管线中的特定部分了, 因为它们运行在GPU上, 所以节省了宝贵的CPU时间, OpenGL着色器是用OpenGL着色器语言(OpenGL Shading Language, GLSL)写成的.
![OpenGL Rendering Pipeline](./images/rendering_pipeline.png)

> Rendering Pipeline, 物体的顶点在最终转化为屏幕坐标之前还会被变换到多个坐标系统(Coordinate System); 将物体的坐标变换到几个过渡坐标系(Intermediate Coordinate System)的优点在于, 在这些特定的坐标系统中, 一些操作或运算更加方便和容易.
![OpenGL Rendering Pipeline](https://learnopengl-cn.github.io/img/01/08/coordinate_systems.png)
