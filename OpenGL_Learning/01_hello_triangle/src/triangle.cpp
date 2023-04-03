/**
 * @file hello_triangle.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief a example that geometric triangle rendering with OpenGL.
 * @version 0.1
 * @date 2023-02-24
 *
 * @copyright Copyright (c) 2023
 *
 */

/* 请确认是在包含GLFW的头文件之前包含了GLAD的头文件.
GLAD的头文件包含了正确的OpenGL头文件(例如GL/gl.h),
所以需要在其它依赖于OpenGL的头文件之前包含GLAD.
---------------------------------------- */
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// --------------------------------------
int main(int argc, char const *argv[])
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const unsigned int SCR_WIDTH = 800;
    const unsigned int SCR_HEIGHT = 600;
    const char *titleName = "HelloOpenGL";
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
                                          titleName, nullptr, nullptr);
    if (window == nullptr)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    /* 顶点数组对象:Vertex Array Object,VAO
    顶点缓冲对象:Vertex Buffer Object,VBO
    元素缓冲对象:Element Buffer Object,EBO 或 索引缓冲对象 Index Buffer Object,IBO
    -------------------------------------------------------------------------
    OpenGL 中图元(Primitive), 任何一个绘制指令的调用都将把图元传递给OpenGL:
        GL_POINTS || GL_TRIANGLES || GL_LINE_STRIP
    ----------------------------------------------- */

    /* set up vertex data (and buffer(s)) and configure vertex attributes
    the three vertices of triangle in OpenGL Normalized Device Coordinates.
    OpenGL仅当3D坐标在3个轴（x、y和z）上 [-1.0, 1.0] 的范围内时才处理它;
    所有在这个范围内的坐标叫做标准化设备坐标(Normalized Device Coordinates),
    此范围内的坐标最终显示在屏幕上, 在这个范围以外的坐标则不会显示.
    -------------------------------------------------------- */
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    /* OpenGL 中 Vertex data 如何在 GPU 显存中进行布局 Layout.
    使用 glVertexAttribPointer 函数告诉 OpenGL 该如何解析顶点数据,
    并应用到逐个顶点属性上; 使用 glEnableVertexAttribArray, 
    以顶点属性位置值作为参数, 启用顶点属性; 顶点属性默认是禁用的.
    ------------------------------------------------------------- */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);

    /* Vertex Shader source code in modern OpenGL
    --------------------------------------------- */
    const char *vertexShaderSource = "#version 460 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "void main()\n"
        "{\n"
        "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
        "}\n";
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    // check for shader compile error.
    int shader_flag;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &shader_flag);
    if (!shader_flag)
    {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cout << "====Error::Shader::Vertex::Compileation_Failed\n" 
                << infoLog << std::endl;
    }

    /* Fragment(pixel) Shader source code in modern OpenGL
    ------------------------------------------------------- */
    const char *fragmentShaderSource = "#version 460 core\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
            // RGBA color mode
        "   FragColor = vec4(1.0f, 0.5f, 0.3f, 1.0f);\n"
        "}\n";
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    // check for shader compile error.
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &shader_flag);
    if (!shader_flag)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cout << "====Error::Shader::Vertex::Compileation_Failed\n" 
                << infoLog << std::endl;
    }

    /* 着色器程序对象 Shader Program Object to Link shaders
    ----------------------------------------------------- */
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors.
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &shader_flag);
    if (!shader_flag)
    {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cout << "====Error::Shader::Program::Linking_Failed\n" 
                << infoLog << std::endl;
    }
    /* 得到的一个程序对象, 调用 glUseProgram 函数, 用刚创建的程序对象作为它的参数,
    以激活这个程序对象; glUseProgram 函数调用之后,
    每个着色器调用和渲染调用都会使用这个程序对象;
    在把着色器对象链接到程序对象以后, 记得删除着色器对象,不再需要它们了.
    --------------------------------------------------------- */
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    /*  */

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}
