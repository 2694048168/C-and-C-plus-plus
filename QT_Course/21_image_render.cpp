/**
 * @file 21_image_render.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-21
 * 
 * @copyright Copyright (c) 2024
 * 
 * 如何将OpenGL渲染的图片保存到本地（正常渲染和离屏渲染）
 * ? https://blog.csdn.net/hust_sheng/article/details/75268056
 * 
 */

#include "src/ImageRender.h"

#include <QApplication>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    ImageRender window;
    window.show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
