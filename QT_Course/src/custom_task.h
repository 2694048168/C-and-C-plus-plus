#ifndef __CUSTOM_TASK_H__
#define __CUSTOM_TASK_H__

#include <QObject>

class MyTask : public QObject
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : explicit MyTask(QObject *parent = nullptr);
    ~MyTask();

    // 工作函数
    void working();

signals:
    // 自定义信号, 传递数据
    void curNumber(int num);
};
#endif // __CUSTOM_TASK_H__
