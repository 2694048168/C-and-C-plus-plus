#include "mainwindow_event.h"

#include "ui_mainwindow_event.h"

#include <QCloseEvent>
#include <QMessageBox>
#include <QResizeEvent>
#include <iostream>

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

void MainWindow::closeEvent(QCloseEvent *ev)
{
    QMessageBox::Button btn = QMessageBox::question(this, "关闭窗口", "您确定要关闭窗口吗?");
    if (btn == QMessageBox::Yes)
    {
        // 接收并处理这个事件
        ev->accept();
    }
    else
    {
        // 忽略这个事件
        ev->ignore();
    }
}

void MainWindow::resizeEvent(QResizeEvent *ev)
{
    std::cout << "resizeEvent callback\n";
}
