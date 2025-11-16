#include "QtVTKApp.h"

#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QtVTKApp window;
    window.show();

    return app.exec();
}
