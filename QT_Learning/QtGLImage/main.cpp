#include "QtGLImage.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QtGLImage w;
    w.show();
    return a.exec();
}
