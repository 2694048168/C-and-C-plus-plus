#include "TCPAssistantClient.h"

#include <QtWidgets/QApplication>

// =================================
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    TCPAssistantClient window_client;
    window_client.show();

    return app.exec();
}
