/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <qapplication.h>
#include <qpushbutton.h>

// ------------------------------
int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    QPushButton hello("Hello from CMake Best Practices!", 0);

    hello.resize(250, 30);
    hello.show();

    return app.exec();
}