#include "FileSystemWatcher.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    FileSystemWatcherWidget window;
    window.show();

    return app.exec();
}
