#include "TCPAssistantServer.h"
#include "TCPAssistantClient.h"

#include <QtWidgets/QApplication>

// =================================
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    TCPAssistantServer window_server;
    window_server.show();

    TCPAssistantClient window_client;
    window_client.show();

    return app.exec();
}
