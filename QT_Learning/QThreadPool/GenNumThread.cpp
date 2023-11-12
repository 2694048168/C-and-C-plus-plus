#include "GenNumThread.h"
#include <QDebug>


GenNumThread::GenNumThread(QObject* parent) :QThread(parent)
{
}

void GenNumThread::run()
{
    qDebug() << "当前线程对象的地址: " << QThread::currentThread();

    int num = 0;
    while (1)
    {
        emit curNumber(num++);
        if (num == 1000)
        {
            break;
        }
        QThread::usleep(1);
    }
    qDebug() << "run() 执行完毕, 子线程退出...";
}
