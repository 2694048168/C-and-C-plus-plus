#include "SerialAssistant.h"

#include <QtWidgets/QApplication>

// =================================
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    SerialAssistant window;
    window.show();

    return app.exec();
}
