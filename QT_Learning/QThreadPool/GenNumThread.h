#pragma once

#include <QThread>

class GenNumThread : public QThread
{
    Q_OBJECT

public:
    explicit GenNumThread(QObject* parent = nullptr);

protected:
    void run() override;
signals:
    // 自定义信号, 传递数据
    void curNumber(int num);
};



