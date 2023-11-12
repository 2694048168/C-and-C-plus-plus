#pragma once

#include <QObject>

class MyThread :public QObject
{
    Q_OBJECT

public:
    explicit MyThread(QObject* parent = nullptr);

    // 工作函数
    void working();
    //void readImgFolder();

signals:
    void curNumber(int num);

};

