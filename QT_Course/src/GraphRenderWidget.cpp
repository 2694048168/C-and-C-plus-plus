#include "GraphRenderWidget.h"

#include <cstddef>

unsigned int VBO, VAO;
unsigned int shaderProgram;

float vertices[] = {
    -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, -0.5f, 0.0f,
};

const char *vertexShaderSource
    = "#version 450 core\n"
      "layout (location=0) in vec3 aPos;\n"
      "void main()\n"
      "{\n"
      "gl_Position = vec4(aPox.x, aPox.y, aPox.z, 1.0);\n"
      "}\n\0";

const char *fragmentShaderSource
    = "#version 450 core\n"
      "out vec4 FragColor;\n"
      "void main()\n"
      "{\n"
      "FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0);\n"
      "}\n\0";

GraphRenderWidget::GraphRenderWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
}

GraphRenderWidget::~GraphRenderWidget() {}

void GraphRenderWidget::initializeGL()
{
    // 从显卡的驱动中加载OpenGL的函数指针
    initializeOpenGLFunctions();

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void GraphRenderWidget::resizeGL(int w, int h) {}

void GraphRenderWidget::paintGL()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
