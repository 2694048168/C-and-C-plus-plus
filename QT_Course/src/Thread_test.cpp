#include "Thread_test.h"

#include "custom_qthread.h"
#include "ui_Thread_test.h"

#include <QDebug>

ThreadTest::ThreadTest(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::CThreadTest)
{
    ui->setupUi(this);

    qDebug() << "主线程对象地址:  " << QThread::currentThread();
    // 创建子线程
    MyThread *subThread = new MyThread;

    connect(subThread, &MyThread::curNumber, this, [=](int num) { ui->label->setNum(num); });

    connect(ui->btn_start, &QPushButton::clicked, this,
            [=]()
            {
                // 启动子线程
                subThread->start();
            });
}

ThreadTest::~ThreadTest()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}
