#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "mythread.h"
#include <QThreadPool>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 1. 创建任务类对象
    Generate* gen = new Generate;
    BubbleSort* bubble = new BubbleSort;
    QuickSort* quick = new QuickSort;

    connect(this, &MainWindow::starting, gen, &Generate::recvNum);
    // 2. 启动子线程
    connect(ui->start, &QPushButton::clicked, this, [=]()
        {
            emit starting(10000);
            QThreadPool::globalInstance()->start(gen);
        });
    connect(gen, &Generate::sendArray, bubble, &BubbleSort::recvArray);
    connect(gen, &Generate::sendArray, quick, &QuickSort::recvArray);
    // 接收子线程发送的数据
    connect(gen, &Generate::sendArray, this, [=](QVector<int> list) {
        QThreadPool::globalInstance()->start(bubble);
        QThreadPool::globalInstance()->start(quick);
        for (int i = 0; i < list.size(); ++i)
        {
            ui->randList->addItem(QString::number(list.at(i)));
        }
        });
    connect(bubble, &BubbleSort::finish, this, [=](QVector<int> list) {
        for (int i = 0; i < list.size(); ++i)
        {
            ui->bubbleList->addItem(QString::number(list.at(i)));
        }
        });
    connect(quick, &QuickSort::finish, this, [=](QVector<int> list) {
        for (int i = 0; i < list.size(); ++i)
        {
            ui->quickList->addItem(QString::number(list.at(i)));
        }
        });
}

MainWindow::~MainWindow()
{
    delete ui;
}

