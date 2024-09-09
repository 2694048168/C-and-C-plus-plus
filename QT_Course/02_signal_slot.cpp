/**
 * @file 02_signal_slot.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

#include <QApplication> // 应用程序类头文件
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
