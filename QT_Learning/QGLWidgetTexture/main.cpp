#include "QGLWidgetTexture.h"
#include <QtWidgets/QApplication>
#include <QSurfaceFormat>

// -------------------------------
int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);

    QGLWidgetTexture w;
    w.show();
    return a.exec();
}
