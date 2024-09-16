/**
 * @file 18_sine_curve_drawing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ Qt事件介绍与正弦曲线绘制
 * @version 0.1
 * @date 2024-09-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "src/sine_curve.h"

#include <QApplication>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    WidgetSineCurve window;
    window.show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
