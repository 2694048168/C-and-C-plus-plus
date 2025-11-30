#include "QtVTKApp.h"

#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    // 需要在创建 QApplication 之前设置
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());

    QApplication app(argc, argv);

    QtVTKApp window;
    window.show();

    return app.exec();
}
