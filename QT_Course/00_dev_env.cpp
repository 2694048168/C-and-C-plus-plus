/**
 * @file 00_dev_env.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "mainwindow.h" // 生成的窗口类头文件

#include <QApplication> // 应用程序类头文件

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    // 创建窗口类对象
    MainWindow window;
    // 显示窗口
    window.show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
