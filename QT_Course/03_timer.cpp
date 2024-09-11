/**
 * @file 03_timer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在进行窗口程序的处理过程中, 经常要周期性的执行某些操作, 或者制作一些动画效果，
 * 看似比较复杂的问题使用定时器就可以完美的解决这些问题,
 * *Qt中提供了两种定时器方式: 使用Qt中的事件处理函数; Qt中的定时器类 QTimer的使用方法;
 * 只需创建一个QTimer类对象,然后调用其 start() 函数开启定时器,
 * 此后QTimer对象就会周期性的发出 timeout() 信号.
 * https://doc.qt.io/qt-6/qtimer.html
// --------------------------------------------
// 启动或重新启动定时器，超时间隔为msec毫秒。
[slot] void QTimer::start(int msec);
// 设置定时器是否只触发一次, 参数为true定时器只触发一次, 为false定时器重复触发, 默认为false
void QTimer::setSingleShot(bool singleShot);
// --------------------------------------------
 * 
 */

#include <QApplication> // 应用程序类头文件
#include <QTime>
#include <QTimer>
#include <iostream>

class CustomTest : public QObject
{
    Q_OBJECT
public:
    explicit CustomTest(QObject *parent = nullptr)
    {
        pTimer->start(500);

        connect(pTimer, &QTimer::timeout, this,
                [=]()
                {
                    QTime   tm = QTime::currentTime();
                    // 格式化当前得到的系统时间
                    QString tm_str = tm.toString("hh:mm:ss.zzz");
                    // 设置要显示的时间
                    std::cout << "[====]Timer 500 ms: " << tm_str.toStdString() << std::endl;
                });

        connect(pTimer, &QTimer::timeout, this,
                [=]()
                {
                    // 获取2s以后的系统时间, 不创建定时器对象, 直接使用类的静态方法
                    QTimer::singleShot(2000, this,
                                       [=]()
                                       {
                                           QTime   tm = QTime::currentTime();
                                           // 格式化当前得到的系统时间
                                           QString tm_str = tm.toString("hh:mm:ss.zzz");
                                           // 设置要显示的时间
                                           std::cout << "[====]Timer 500 ms: " << tm_str.toStdString() << std::endl;
                                       });
                });
    }

private:
    // 创建定时器对象
    QTimer *pTimer = new QTimer(this);
};

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    // CustomTest test;

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
