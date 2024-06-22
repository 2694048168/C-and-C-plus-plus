/**
 * @file rotating_cube.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Draw Rotating Cube via OpenGL
 * @version 0.1
 * @date 2024-06-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef GLAD_IMPLEMENTATION
#    define GLAD_IMPLEMENTATION
#    include <glad/glad.h>
#endif // GLAD_IMPLEMENTATION

// GLFW_INCLUDE_NONE information
// https://www.glfw.org/docs/latest/build_guide.html
#ifndef GLFW_INCLUDE_NONE
#    define GLFW_INCLUDE_NONE
#    include <GLFW/glfw3.h>
#endif // GLFW_INCLUDE_NONE

// 向量数学计算
#include <glm/glm.hpp>
// 暂未使用
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include <iostream>
#include <string>

// 窗口尺寸
const int WIDTH  = 600;
const int HEIGHT = 600;

// 着色器代码
const char *vertexShaderSource
    = "#version 330 core\n"
      "layout (location = 0) in vec3 aPos;\n"
      "uniform mat4 model;\n"
      "uniform mat4 view;\n"
      "uniform mat4 projection;\n"
      "void main() {\n"
      "   gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
      "}\0";
// 立方体的颜色
const char *fragmentShaderSource
    = "#version 330 core\n"
      "out vec4 FragColor;\n"
      "void main() {\n"
      "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n" // 橙色
      "}\0";

// 创建着色器程序
unsigned int CreateShaderProgram(const char *vertexShaderSource, const char *fragmentShaderSource)
{
    {
        // 创建顶点着色器
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);

        // 检查顶点着色器编译错误
        int  success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // 创建片段着色器
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);

        // 检查片段着色器编译错误
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // 创建着色器程序
        unsigned int shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // 检查着色器程序链接错误
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }

        // 删除着色器对象
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return shaderProgram;
    }
}

// 在 glfw 窗口标题显示中文
void SetWindowTitleUTF8(GLFWwindow *window, const char *title)
{
    int      wlen = MultiByteToWideChar(CP_ACP, 0, title, -1, NULL, 0);
    wchar_t *wstr = new wchar_t[wlen];
    MultiByteToWideChar(CP_ACP, 0, title, -1, wstr, wlen);

    int   utf8len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    char *utf8str = new char[utf8len];
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, utf8str, utf8len, NULL, NULL);

    glfwSetWindowTitle(window, utf8str);

    delete[] wstr;
    delete[] utf8str;
}

// 初始化 GLFW 和 OpenGL
bool InitOpenGL()
{
    // 初始化 GLFW
    if (!glfwInit())
    {
        return false;
    }

    // 设置 GLFW 版本
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 创建窗口
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    SetWindowTitleUTF8(window, "OpenGL 旋转的立方体");

    // 设置当前窗口为活动窗口
    glfwMakeContextCurrent(window);

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    // 设置视口
    glViewport(0, 0, WIDTH, HEIGHT);

    // 启用深度测试
    glEnable(GL_DEPTH_TEST);

    return true;
}

// 绘制立方体边线
void DrawCube(unsigned int shaderProgram, unsigned int VAO)
{
    glBindVertexArray(VAO);

    glDrawArrays(GL_LINE_LOOP, 0, 4);  // 前面
    glDrawArrays(GL_LINE_LOOP, 4, 4);  // 后面
    glDrawArrays(GL_LINE_LOOP, 8, 4);  // 左侧
    glDrawArrays(GL_LINE_LOOP, 12, 4); // 右侧
    glDrawArrays(GL_LINE_LOOP, 16, 4); // 顶部
    glDrawArrays(GL_LINE_LOOP, 20, 4); // 底部

    glBindVertexArray(0);
}

int main()
{
    // 初始化 OpenGL
    if (!InitOpenGL())
    {
        return 1;
    }

    // 创建着色器程序
    unsigned int shaderProgram = CreateShaderProgram(vertexShaderSource, fragmentShaderSource);

    // 立方体顶点
    float vertices[] = {
        // 前面
        -0.65f,
        -0.65f,
        0.65f,
        0.65f,
        -0.65f,
        0.65f,
        0.65f,
        0.65f,
        0.65f,
        -0.65f,
        0.65f,
        0.65f,

        // 后面
        -0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        -0.65f,

        // 左侧
        -0.65f,
        -0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        0.65f,

        // 右侧
        0.65f,
        -0.65f,
        0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        0.65f,
        -0.65f,
        0.65f,
        0.65f,
        0.65f,

        // 顶部
        -0.65f,
        0.65f,
        0.65f,
        0.65f,
        0.65f,
        0.65f,
        0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        0.65f,
        -0.65f,

        // 底部
        -0.65f,
        -0.65f,
        0.65f,
        0.65f,
        -0.65f,
        0.65f,
        0.65f,
        -0.65f,
        -0.65f,
        -0.65f,
        -0.65f,
        -0.65f,
    };

    // 创建 VAO 和 VBO
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // 绑定 VAO 和 VBO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // 将顶点数据复制到 VBO
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 设置顶点属性指针
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // 解除绑定
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // 模型、视图和投影矩阵
    glm::mat4 model      = glm::mat4(1.0f);
    glm::mat4 view       = glm::mat4(1.0f);
    view                 = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);

    // 旋转角度变量
    float angleX = 0;
    float angleY = 0;
    float angleZ = 0;

    // 获取控制台窗口句柄
    HWND   hConsole = GetConsoleWindow();
    HANDLE hStdOut  = GetStdHandle(STD_OUTPUT_HANDLE);

    // 帧率控制
    double timePrevious        = glfwGetTime();
    double timeLast            = 0.0;
    int    fps                 = 0;
    double standard_frame_rate = 60;                        // 目标帧率
    double frameTimeLimit      = 1.0 / standard_frame_rate; // 初始跳帧时间
    float  fltFactor           = 0.0f;
    double timeFramePrevious   = glfwGetTime();
    double timeFrameLast       = 0.0;

    // 启用抗锯齿 (貌似没作用)
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_MULTISAMPLE);
    // 设置线宽
    glLineWidth(1);

    // 主循环
    while (!glfwWindowShouldClose(glfwGetCurrentContext()))
    {
        // 帧率控制逻辑
        timeFramePrevious = glfwGetTime();
        if ((timeFramePrevious - timeFrameLast) >= 1.0f)
        {
            // 定行输出
            COORD       pos = {1, 1};
            DWORD       charsWritten;
            std::string fpsStr = "当前帧率: " + std::to_string(fps) + "   ";
            WriteConsoleOutputCharacterA(hStdOut, fpsStr.c_str(), fpsStr.length(), pos, &charsWritten);
            pos                      = {1, 2};
            std::string frameTimeStr = "跳帧时间: " + std::to_string(frameTimeLimit) + "   ";
            WriteConsoleOutputCharacterA(hStdOut, frameTimeStr.c_str(), frameTimeStr.length(), pos, &charsWritten);

            timeFrameLast = timeFramePrevious;

            // 根据当前帧率调整跳帧时间，让帧率趋近 standard_frame_rate
            if (fps > standard_frame_rate)
            {
                if (fps > (standard_frame_rate * 2))
                {
                    fltFactor = 1;
                }
                else if (fps > (standard_frame_rate * 1.5))
                {
                    fltFactor = 0.5;
                }
                else if (fps > (standard_frame_rate * 1.1))
                {
                    fltFactor = 0.1;
                }
                else
                {
                    fltFactor = 0.01;
                }
                frameTimeLimit = frameTimeLimit + (frameTimeLimit * fltFactor);
            }
            else if (fps < standard_frame_rate)
            {
                if (fps < (standard_frame_rate / 1.1))
                {
                    fltFactor = 0.1;
                }
                else if (fps < (standard_frame_rate / 1.5))
                {
                    fltFactor = 0.5;
                }
                else if (fps < (standard_frame_rate / 2))
                {
                    fltFactor = 1;
                }
                frameTimeLimit = frameTimeLimit - (frameTimeLimit * fltFactor);
            }
            fps = 0;

            // 未避免当前跳帧时间太小（趋近0），导致恢复过慢，设置小于 N 值直接重置为
            // 0.1
            if (frameTimeLimit < 0.000001)
            {
                frameTimeLimit = 0.1;
            }
        }

        timePrevious = timeFramePrevious;
        if ((timePrevious - timeLast) < frameTimeLimit)
        {
            continue;
        }
        timeLast = timePrevious;
        fps += 1;

        // 清除颜色缓冲区
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 使用着色器程序
        glUseProgram(shaderProgram);

        // 每一帧旋转角度，也是速度控制
        angleX += (5.7 * 0.00002);
        angleY += (2.1 * 0.00002);
        angleZ += (9.33 * 0.00002);
        // 创建旋转矩阵
        model = glm::rotate(model, glm::radians(angleX), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(angleY), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(angleZ), glm::vec3(0.0f, 0.0f, 1.0f));

        // 设置模型、视图和投影矩阵
        unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // 绘制立方体
        DrawCube(shaderProgram, VAO);

        // 交换缓冲区
        glfwSwapBuffers(glfwGetCurrentContext());
        // 处理事件
        glfwPollEvents();
    }

    // 删除 VAO, VBO 和着色器程序
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    // 退出 GLFW
    glfwTerminate();

    return 0;
}
