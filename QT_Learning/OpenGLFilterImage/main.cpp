#include <QApplication>
#include <QMainWindow>
#include <QSurfaceFormat>
#include <QDebug>
#include "glpanel.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QSurfaceFormat format = QSurfaceFormat::defaultFormat();
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setVersion(4, 2);
    format.setDepthBufferSize(24);
    format.setAlphaBufferSize(8);
    format.setStencilBufferSize(8);
    QSurfaceFormat::setDefaultFormat(format);

    QMainWindow window;
    window.setWindowTitle("GLFilter");
    window.resize(800, 600);
    GLPanel *panel = new GLPanel(&window);
    window.setCentralWidget(panel);
    window.show();

    return app.exec();
}
