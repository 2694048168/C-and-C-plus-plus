#include "mainwindow.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MainWindow window;
    window.setWindowTitle("Gazer");
    window.show();

    return app.exec();
}
