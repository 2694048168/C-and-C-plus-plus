#include "TCPAssistantServer.h"

#include <QtWidgets/QApplication>

// =================================
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    TCPAssistantServer window_server;
    window_server.show();

    return app.exec();
}
