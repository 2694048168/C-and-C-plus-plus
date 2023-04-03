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

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

const char *vertexShaderSource = "#version 460 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\n";

const char *fragmentShaderSource = "#version 460 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
        // RGBA color mode
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n";


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

    // build and compile our shader program
    // ------------------------------------
    // the Vertex Shader in OpenGL with OpenGL Shader Language(GLSL).
    /* Vertex Shader: GLSL source code ---> create shader object(ID index) 
    ---> shader object load GLSL source ---> compile shader and judage */
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

    // the Fragment Shader in OpenGL with OpenGL Shader Language(GLSL).
    /* Fragment Shader: GLSL source code ---> create shader object(ID index) 
    ---> shader object load GLSL source ---> compile shader and judage */
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    // check for shader compile error.
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &shader_flag);
    if (!shader_flag)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cout << "====Error::Shader::Fragment::Compileation_Failed\n"
                << infoLog << std::endl;
    }

    // the Shader Program that running on GPU device in OpenGL.
    /* the Shader Program Object in OpenGL is linking 
    the multi-shader(vertex shader and fragment shader):
    create shader program object(ID index) ---> 
    link shader into this object by order and judage --->
    use or activate this program object ---> 
    delete all shader object(vertex and fragment) */
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
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    /* set up vertex data (and buffer(s)) and configure vertex attributes
    the three vertices of triangle in OpenGL Normalized Device Coordinates.
    OpenGL仅当3D坐标在3个轴（x、y和z）上 [-1.0, 1.0] 的范围内时才处理它;
    所有在这个范围内的坐标叫做标准化设备坐标(Normalized Device Coordinates),
    此范围内的坐标最终显示在屏幕上, 在这个范围以外的坐标则不会显示.
    -------------------------------------------------------- */
    float vertices[] = {
        -0.5f,  0.5f, 0.0f, /* top right */
         0.5f, -0.5f, 0.0f, /* bottom right */
        -0.0f, -0.5f, 0.0f, /* bottom left */
        -0.0f,  0.0f, 0.0f /* top left */
    };
    unsigned int  indices[] = { /* note that we start from 0. */
        0, 1, 3, /* first Triangle */
        1, 2, 3 /* second Triangle */
    };

    // the Vertex Buffer Object(VBO) in OpenGL for GPU memory.
    // the Vertex Array Object(VAO)
    // the Element Buffer Object(VAO) or Index Buffer Object(IBO)
    /* VBO: create ID index ---> generate corresponding buffer 
    ---> bind corresponding buffer ---> copy data into corresponding buffer
    ---> configure vertex attributes for OpenGL to parse in GPU memory */
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    /* bind the Vertex Array Object first, then bind and set vertex buffer(s), 
    and then configure vertex attributes(s). */
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    /* note that this is allowed, the call to glVertexAttribPointer registered 
    VBO as the vertex attribute's bound vertex buffer object so afterwards 
    we can safely unbind */
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    /* remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object 
    IS stored in the VAO; keep the EBO bound. */
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    /* You can unbind the VAO afterwards so other VAO calls won't accidentally 
    modify this VAO, but this rarely happens. Modifying other VAOs requires 
    a call to glBindVertexArray anyways, so we generally don't unbind VAOs
    (nor VBOs) when it's not directly necessary. */
    glBindVertexArray(0); 

    // uncomment this call to draw in wireframe polygons.
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        processInput(window);

        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO); /* seeing as we only have a single VAO 
        there's no need to bind it every time, but we'll do so to keep 
        things a bit more organized */
        // glDrawArrays(GL_TRIANGLES, 0, 6);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        // glBindVertexArray(0); /* no need to unbind it every time */
 
        /* glfw: swap buffers and poll IO events 
        (keys pressed/released, mouse moved etc.) */
        // ------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
