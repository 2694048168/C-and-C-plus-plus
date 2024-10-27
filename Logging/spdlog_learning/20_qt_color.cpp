/**
 * @file 20_qt_color.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "20_mainwidget.hpp"

#include <QApplication>

// -----------------------------------
int main(int argc, char **argv)
{
    /* 21. Log to Qt with nice colors
    * 在使用 spdlog 与 Qt 框架集成时, 可以将日志消息直接输出到 Qt 的 QTextEdit 控件中,
    * 并且可以使用颜色来区分不同级别的日志消息. 
    * 这对于开发调试和实时监控非常有用, 因为它能够提供直观的日志信息展示.
    * 
    * @note: Custom sink for QPlainTextEdit or QTextEdit and its children
    *  (QTextBrowser... etc) Building and using requires Qt library.
    * @Warning: the qt_sink won't be notified if the target widget is destroyed.
    * If the widget's lifetime can be shorter than the logger's one, 
    * you should provide some permanent QObject, and then use a standard signal/slot.
    */

    QApplication app(argc, argv);

    // 创建并显示主窗口
    MainWindow window;
    window.show();

    return app.exec();
}
