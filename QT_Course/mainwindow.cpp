#include "mainwindow.h"

#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CMainWindow) // 基于mainwindow.ui创建一个实例对象
{
    // 将 mainwindow.ui 的实例对象和 当前类的对象进行关联
    // 这样同名的连个类对象就产生了关联, 合二为一了
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
