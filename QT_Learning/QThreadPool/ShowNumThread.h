#pragma once

#include <QThread>
#include <QVector>

class ShowNumThread : public QThread
{
    Q_OBJECT

public:
    explicit ShowNumThread(QObject* parent = nullptr);

    void reciveNum(int num);

protected:
    void run() override;
signals:
    void sendArray(QVector<int> vec);

private:
    int m_num;

};

