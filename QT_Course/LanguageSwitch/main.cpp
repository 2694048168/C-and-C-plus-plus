/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-28
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "mainwindow.h"

#include <QApplication>

//  -----------------------------
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();
}
