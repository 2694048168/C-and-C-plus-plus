#pragma once

#include <QtWidgets/QWidget>
#include "ui_QThreadPool.h"
#include "ShowNumThread.h"
#include "GenNumThread.h"

QT_BEGIN_NAMESPACE
namespace Ui { class QThreadPoolClass; };
QT_END_NAMESPACE

class QThreadPool : public QWidget
{
    Q_OBJECT

public:
    QThreadPool(QWidget* parent = nullptr);
    ~QThreadPool();

private slots:
    void sl_showNum();
signals:
    void starting(int num);
private:
    Ui::QThreadPoolClass* ui;
    ShowNumThread* random_num_thread;
    GenNumThread* subThread;
};
