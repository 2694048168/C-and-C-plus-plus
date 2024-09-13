/**
 * @file 08_widget_container.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt中常用的容器控件, 包括: Widget, Frame, Group Box, Scroll Area, 
 * Tool Box, Tab Widget, Stacked Widget
 * 
 * ? https://subingwen.cn/qt/qt-containers/
 * 
 */

#include <QApplication>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    std::cout << "=========== Qt中常用的容器控件 ===============\n";
    std::cout << "==========================\n";

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
