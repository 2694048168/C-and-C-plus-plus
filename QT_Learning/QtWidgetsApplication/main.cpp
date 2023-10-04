#include "QtWidgetsApplication.h"
#include "newmenuwidget.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    //QtWidgetsApplication window;
    //window.show();

    newMenuWidget window;
    window.show();

    return app.exec();
}
