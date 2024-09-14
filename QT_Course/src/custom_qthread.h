#ifndef __CUSTOM_THREAD_H__
#define __CUSTOM_THREAD_H__

#include <QThread>

class MyThread : public QThread
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : explicit MyThread(QObject *parent = nullptr);
    ~MyThread();

protected:
    void run() override;

signals:
    // 自定义信号, 传递数据
    void curNumber(int num);
};
#endif // __CUSTOM_THREAD_H__
