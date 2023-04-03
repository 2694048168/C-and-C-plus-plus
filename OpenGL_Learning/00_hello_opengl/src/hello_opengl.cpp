/**
 * @file hello_opengl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief a simple example that geometric graphics rendering with OpenGL.
 * @version 0.1
 * @date 2023-02-23
 *
 * @copyright Copyright (c) 2023
 *
 */

/* the 'GLAD' headfile must be front of 'GLFW' headfile. */
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

/* 当用户改变窗口的大小的时候, viewpoint 视口也应该被调整;
对窗口注册一个回调函数(Callback Function),它会在每次窗口大小被调整的时候被调用 */
void framebuffer_size_callback(GLFWwindow *window, int width, int height);

/* 希望能够在GLFW中实现一些输入控制,这可以通过使用GLFW的几个输入函数来完成;
将会使用GLFW的glfwGetKey函数,它需要一个窗口以及一个按键作为输入;
这个函数将会返回这个按键是否正在被按下 */
void processInput_callback(GLFWwindow *window);

// ------------------------------------
int main(int argc, char const *argv[])
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();                                    /* init GLFW */
    /* Linux 上运行 glxinfo
    Windows 上使用 OpenGL Extension Viewer 工具 */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); /* OpenGL Major version=4 */
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6); /* OpenGL Minor version=6 */
    /* OpenGL Profile=core */
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    /* if Mac OS, it need to configure. */
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    /* glfw window creation
    ----------------------------------- */
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
    /* 创建完窗口我们就可以通知GLFW将窗口的上下文设置为当前线程的主上下文 */
    glfwMakeContextCurrent(window);
    /* 需要注册这个函数,告诉 GLFW 希望每当窗口调整大小的时候调用这个函数 */
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    /* glad: load all OpenGL function pointers
    GLAD是用来管理OpenGL的函数指针的,
    所以在调用任何OpenGL的函数之前我们需要初始化GLAD.
    GLFW给的是glfwGetProcAddress, 它根据编译的系统定义了正确的函数.
    -------------------------------------------------------- */
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    /* 渲染循环 Render Loop
    ------------------------------------- */
    while (!glfwWindowShouldClose(window))
    {
        /* input */
        processInput_callback(window);

        /* render */
        /* 在每个新的渲染迭代开始的时候总是希望清屏,否则仍能看见上一次迭代的渲染结果;
        可以通过调用glClear函数来清空屏幕的颜色缓冲,它接受一个缓冲位(Buffer Bit)
        来指定要清空的缓冲: 
        GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
        由于现在只关心颜色值, 所以只清空颜色缓冲;

        注意, 除了glClear之外, 还调用了glClearColor来设置清空屏幕所用的颜色;
        当调用glClear函数, 清除颜色缓冲之后, 整个颜色缓冲都会被填充为glClearColor里所设置的颜色;
        在 OpenGL 中, glClearColor 函数是一个状态设置函数;
        而 glClear 函数则是一个状态使用的函数, 它使用了当前的状态来获取应该清除为的颜色.
        ------------------------------------------------------------------- */
        glClearColor(0.1f, 0.3f, 0.5f, 1.0f); /* RGBA color mode */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Double buffer(front buffer and back buffer) and then swap.
        双缓冲(Double Buffer)
        应用程序使用单缓冲绘图时可能会存在图像闪烁的问题;
        这是因为生成的图像不是一下子被绘制出来的,而是按照从左到右,由上而下逐像素地绘制而成的;
        最终图像不是在瞬间显示给用户,而是通过一步一步生成的,这会导致渲染的结果很不真实;
        为了规避这些问题，应用双缓冲渲染窗口应用程序;
        前缓冲保存着最终输出的图像, 它会在屏幕上显示;而所有的的渲染指令都会在后缓冲上绘制;
        当所有的渲染指令执行完毕后, 交换(Swap)前缓冲和后缓冲这样图像就立即呈显出来
        ---------------------------------------------------------------- */
        glfwSwapBuffers(window);

        /* glfwPollEvents函数检查有没有触发什么事件
        比如键盘输入、鼠标移动等、更新窗口状态, 并调用对应的回调函数 */
        glfwPollEvents();
    }

    /* glfw: terminate, clearing all previously allocated GLFW resources.
    当渲染循环结束后我们需要正确释放/删除之前的分配的所有资源
    ------------------------------------------------ */
    glfwTerminate();

    return 0;
}

/* process all input: query GLFW whether relevant keys are pressed/released
 this frame and react accordingly */
void processInput_callback(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

/* glfw: whenever the window size changed (by OS or user resize)
 this callback function executes */
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    /* make sure the viewport matches the new window dimensions;
    note that width and height will be significantly larger than 
    specified on retina displays. 
    开始渲染之前还有一件重要的事情要做, 必须告诉OpenGL渲染窗口的尺寸大小
    即视口(Viewport), 这样OpenGL才只能知道怎样根据窗口大小显示数据和坐标.
    ------------------------------------------------------------- */
    glViewport(0, 0, width, height);
}
