#include <QApplication>
#include <QMainWindow>

#include "BlobToolControl.h"

// --------------------------------
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QMainWindow window;

    BlobToolControl *pBlobToolWidget = new BlobToolControl(&window);

    window.setCentralWidget(pBlobToolWidget);
    window.show();

    return app.exec();
}
