/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "AddParameter.h"

#include <QtWidgets/QApplication>

// -------------------------------
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    AddParamWidget window;
    window.resize(860, 640);
    window.show();

    // 示例：动态使用参数
    QObject::connect(&window, &AddParamWidget::destroyed,
                     [&]()
                     {
                         // 获取所有参数
                         auto params = window.getAllParameters();

                         // 使用特定参数
                         if (window.getParameterValue("width").isValid())
                         {
                             int width = window.getParameterValue("width").toInt();
                             qDebug() << "Using width parameter:" << width;
                         }

                         // 修改参数值
                         window.setParameterValue("threshold", 0.75);
                     });

    return app.exec();
}
