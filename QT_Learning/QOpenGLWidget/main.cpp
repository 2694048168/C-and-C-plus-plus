#include "QOpenGLWidget.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QOpenGLWidget w;
    w.show();
    return a.exec();
}
