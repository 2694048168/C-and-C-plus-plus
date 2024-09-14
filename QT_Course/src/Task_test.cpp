#include "Task_test.h"

#include "custom_task.h"
#include "ui_Task_test.h"

#include <QDebug>
#include <QThread>

TaskTest::TaskTest(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::CThreadTest)
{
    ui->setupUi(this);

    qDebug() << "主线程对象地址:  " << QThread::currentThread();
    // 创建线程对象
    QThread *sub = new QThread;
    // 创建工作的类对象
    // 千万不要指定给创建的对象指定父对象
    // 如果指定了: QObject::moveToThread: Cannot move objects with a parent
    MyTask  *work = new MyTask;
    // 将工作的类对象移动到创建的子线程对象中
    work->moveToThread(sub);
    // 启动线程
    sub->start();
    // 让工作的对象开始工作, 点击开始按钮, 开始工作
    connect(ui->btn_start, &QPushButton::clicked, work, &MyTask::working);
    // 显示数据
    connect(work, &MyTask::curNumber, this, [=](int num) { ui->label->setNum(num); });
}

TaskTest::~TaskTest()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}
