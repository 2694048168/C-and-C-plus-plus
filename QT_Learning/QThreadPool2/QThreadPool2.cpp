#include "QThreadPool2.h"

QThreadPool2::QThreadPool2(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QThreadPool2Class())
{
    ui->setupUi(this);
    // 1. creaate sub-thread
    sub_thread = new QThread;
    // 2. create task-class instance
    task = new MyThread;
    // 3. move task-class instance into sub-thread
    task->moveToThread(sub_thread);

    // 4. 启动线程
    sub_thread->start();
    // 5. 让工作的对象开始工作, 点击开始按钮, 开始工作
    connect(ui->startBtn, &QPushButton::clicked, task, &MyThread::working);
    // 显示数据
    connect(task, &MyThread::curNumber, this, [=](int num)
        {
            ui->num_label->setNum(num);
        });

    // 释放内存资源
    connect(this, &QThreadPool2::destroyed, this, [=]() {
        sub_thread->quit();
        sub_thread->wait();
        sub_thread->deleteLater(); // delete sub_thread

        task->deleteLater(); // delete sub_thread
        });
}

QThreadPool2::~QThreadPool2()
{
    delete ui;
}
