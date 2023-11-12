#include "QThreadPool.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // 注册 qt 不支持的(自定义的数据结构), 
    // 利用 主/子线程的信号槽机制传递数据
    qRegisterMetaType<QVector<int>>("QVector<int>");
    QThreadPool window;
    window.show();

    return app.exec();
}
