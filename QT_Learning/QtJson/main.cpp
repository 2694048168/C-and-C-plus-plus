#include "QtJson.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QtJson w;
    w.show();
    return a.exec();
}
