#include "QThreadPool.h"
#include <QDebug>

QThreadPool::QThreadPool(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QThreadPoolClass())
{
    ui->setupUi(this);

    // step 1. create sub-thread
    random_num_thread = new ShowNumThread;
    connect(this, &QThreadPool::starting, random_num_thread, &ShowNumThread::reciveNum);
    // step 2. start sub-thread
    connect(ui->startBtn, &QPushButton::clicked, this, [=]() {
        emit starting(10000);
        random_num_thread->start();
        });
    // step 3. receive sub-thread data
    connect(random_num_thread, &ShowNumThread::sendArray, this, [=](QVector<int> vec) {
        for (size_t i = 0; i < vec.size(); ++i)
            ui->rand_listWidget->addItem(QString::number(vec.at(i)));
        });

#if true
    qDebug() << "主线程对象地址:  " << QThread::currentThread();
    // 创建子线程
    subThread = new GenNumThread;

    connect(subThread, &GenNumThread::curNumber, this, [=](int num)
        {
            ui->num_label->setNum(num);
        });

    connect(ui->startBtn, &QPushButton::clicked, this, [=]()
        {
            // 启动子线程
            subThread->start();
        });
#else
    connect(ui->startBtn, &QPushButton::clicked, this, &QThreadPool::sl_showNum);
#endif
}

QThreadPool::~QThreadPool()
{
    delete ui;
}

void QThreadPool::sl_showNum()
{
    int num = 0;
    while (true)
    {
        ++num;
        if (num == 10000000)
            break;

        ui->num_label->setNum(num);
    }
}
