## the OpenGL series by The Cherno

![rendering_pipeline_OpenGL](./rendering_pipeline.png)

### the organization of project

```
. ComputerGraphics
|—— 00_opengl
|   |—— src
|   |—— |—— opengl.cpp
|   |—— CMakeLists.txt
|—— 01_triangle
|   |—— src
|   |—— |—— triangle.cpp
|   |—— CMakeLists.txt
|—— 02_shader
|   |—— res
|   |—— |—— shaders
|   |—— |—— |—— Basic.shader
|   |—— src
|   |—— |—— Application.cpp
|   |—— CMakeLists.txt
|—— external
|   |—— GLEW
|   |—— |—— include
|   |—— |—— src
|   |—— |—— bin
|   |—— GLFW
|   |—— |—— include
|   |—— |—— lib-mingw-w64
|   |—— |—— lib-vc2022
|—— CMakeLists.txt
|—— rendering_pileline.png
|—— README.md
```

### Overview of OpenGL Series

- Welcome to OpenGL
- Setting up OpenGL and Creating a Window in C++
- Using Modern OpenGL in C++
- Vertex Buffers and Drawing a Triangle in OpenGL
- Vertex Attributes and Layouts in OpenGL
- How Shaders Work in OpenGL
- Writting a Shader in OpenGL
- How I Deal with Shaders in OpenGL
- Index Buffers in OpenGL
- Dealing with Errors in OpenGL
- Uniforms in OpenGL
- Vertex Arrays in OpenGL
- Abstracting OpenGL into Classes
- Buffer Layout Abstraction in OpenGL
- Shader Abstraction in OpenGL
- Writing a Basic Renderer in OpenGL
- Textures in OpenGL
- Blending in OpenGL
- Maths in OpenGL
- Projection Matrices in OpenGL
- Model View Projection Matrices in OpenGL
- ImGui in OpenGL
- Rendering Multiple Objects in OpenGL
- Setting up a Test Framework for OpenGL
- Creating Tests in OpenGL
- Creating a Texture Test in OpenGL
- How to make your UNIFORMS FASTER in OpenGL
- An Introduction of Batch Rendering 
- Colors of Batch Rendering
- Textures of Batch Rendering
- Dynamic Geometry of Batch Rendering
