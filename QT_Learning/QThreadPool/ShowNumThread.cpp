#include "ShowNumThread.h"
#include <QRandomGenerator>
#include <QElapsedTimer>
#include <QDebug>

ShowNumThread::ShowNumThread(QObject* parent) :QThread(parent)
{

}

void ShowNumThread::reciveNum(int num)
{
    m_num = num;
}

void ShowNumThread::run()
{
    qDebug() << "生成随机数的线程地址 " << QThread::currentThread();
    QVector<int> vec;
    QElapsedTimer timer;
    timer.start();
    for (size_t i = 0; i < m_num; ++i)
    {
        vec.push_back(QRandomGenerator::global()->bounded(0, 100000));
    }
    int milsec = timer.elapsed();
    qDebug() << "生成 " << m_num << " 个随机数总用时：" << milsec << " 毫秒";

    emit sendArray(vec);
}
