#pragma once

#include <QtWidgets/QWidget>
#include "ui_QThreadPool2.h"
#include "MyThread.h"
#include <QThread>


QT_BEGIN_NAMESPACE
namespace Ui { class QThreadPool2Class; };
QT_END_NAMESPACE

class QThreadPool2 : public QWidget
{
    Q_OBJECT

public:
    QThreadPool2(QWidget* parent = nullptr);
    ~QThreadPool2();

private:
    Ui::QThreadPool2Class* ui;
    MyThread* task;
    QThread* sub_thread;
};
