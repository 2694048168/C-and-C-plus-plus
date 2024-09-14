/**
 * @file 13_thread_task.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在进行桌面应用程序开发的时候,假设应用程序在某些情况下需要处理比较复杂的逻辑,
 * 如果只有一个线程去处理,就会导致窗口卡顿,无法处理用户的相关操作.
 * 这种情况下就需要使用多线程,其中一个线程处理窗口事件,其他线程进行逻辑运算,
 * 多个线程各司其职,不仅可以提高用户体验还可以提升程序的执行效率.
 * ?在qt中使用了多线程,有些事项是需要额外注意的:
 * *1. 默认的线程在Qt中称之为窗口线程,也叫主线程,负责窗口事件处理或者窗口控件数据的更新;
 * *2. 子线程负责后台的业务逻辑处理,子线程中不能对窗口对象做任何操作,这些事情需要交给窗口线程处理;
 * *3. 主线程和子线程之间如果要进行数据的传递,需要使用Qt中的信号槽机制;
 * 
 * 1. 线程类 QThread
 * ?Qt中提供了一个线程类,通过这个类就可以创建子线程了,Qt中一共提供了两种创建子线程的方式.
1.1 常用共用成员函数
// QThread 类常用 API
// 构造函数
QThread::QThread(QObject *parent = Q_NULLPTR);
// 判断线程中的任务是不是处理完毕了
bool QThread::isFinished() const;
// 判断子线程是不是在执行任务
bool QThread::isRunning() const;

// Qt中的线程可以设置优先级
// 得到当前线程的优先级
Priority QThread::priority() const;
void QThread::setPriority(Priority priority);
优先级:
    QThread::IdlePriority         --> 最低的优先级
    QThread::LowestPriority
    QThread::LowPriority
    QThread::NormalPriority
    QThread::HighPriority
    QThread::HighestPriority
    QThread::TimeCriticalPriority --> 最高的优先级
    QThread::InheritPriority      --> 子线程和其父线程的优先级相同, 默认是这个
// 退出线程, 停止底层的事件循环
// 退出线程的工作函数
void QThread::exit(int returnCode = 0);
// 调用线程退出函数之后, 线程不会马上退出因为当前任务有可能还没有完成, 调回用这个函数是
// 等待任务完成, 然后退出线程, 一般情况下会在 exit() 后边调用这个函数
bool QThread::wait(unsigned long time = ULONG_MAX);
// ==================================================
1.2 信号槽
// 和调用 exit() 效果是一样的
// 代用这个函数之后, 再调用 wait() 函数
[slot] void QThread::quit();
// 启动子线程
[slot] void QThread::start(Priority priority = InheritPriority);
// 线程退出, 可能是会马上终止线程, 一般情况下不使用这个函数
[slot] void QThread::terminate();

// 线程中执行的任务完成了, 发出该信号
// 任务函数中的处理逻辑执行完毕了
[signal] void QThread::finished();
// 开始工作之前发出这个信号, 一般不使用
[signal] void QThread::started();
// ==================================================
1.3 静态函数
// 返回一个指向管理当前执行线程的QThread的指针
[static] QThread *QThread::currentThread();
// 返回可以在系统上运行的理想线程数 == 和当前电脑的 CPU 核心数相同
[static] int QThread::idealThreadCount();
// 线程休眠函数
[static] void QThread::msleep(unsigned long msecs);	// 单位: 毫秒
[static] void QThread::sleep(unsigned long secs);	// 单位: 秒
[static] void QThread::usleep(unsigned long usecs);	// 单位: 微秒
// ==================================================
1.4 任务处理函数
// 子线程要处理什么任务, 需要写到 run() 中
[virtual protected] void QThread::run();
// ==================================================
 * *这个run()是一个虚函数,如果想让创建的子线程执行某个任务,需要写一个子类让其继承QThread,
 * 并且在子类中重写父类的run()方法, 函数体就是对应的任务处理流程.
 * ?这个函数是一个受保护的成员函数,不能够在类的外部调用;如果想要让线程执行这个函数中的业务流程,
 * 需要通过当前线程对象调用槽函数start()启动子线程, 当子线程被启动,这个run()函数也就在线程内部被调用了.
 * 
 * 2. 使用方式1
 * ----1. 需要创建一个线程类的子类，让其继承QT中的线程类 QThread;
 * ----2. 重写父类的 run() 方法，在该函数内部编写子线程要处理的具体的业务流程;
 * ----3. 在主线程中创建子线程对象，new 一个就可以了;
 * ----4. 启动子线程, 调用 start() 方法;
 * ?当子线程别创建出来之后, 父子线程之间的通信可以通过信号槽的方式,注意事项:
 * *1. 在Qt中在子线程中不要操作程序中的窗口类型对象, 不允许, 如果操作了程序就挂了;
 * *2. 只有主线程才能操作程序中的窗口对象, 默认的线程就是主线程,自己创建的就是子线程;
 * 
 * !这种在程序中添加子线程的方式是非常简单的,但是也有弊端,
 * 假设要在一个子线程中处理多个任务,所有的处理逻辑都需要写到run()函数中,
 * 这样该函数中的处理逻辑就会变得非常混乱,不太容易维护.
 * 
 * 3. 使用方式2
 * ----1. 创建一个新的类，让这个类从QObject派生;
 * ----2. 在这个类中添加一个公共的成员函数worker，函数体就是子线程中执行的业务逻辑;
 * ----3. 在主线程中创建一个QThread对象, 这就是子线程的对象;
 * ----4. 在主线程中创建工作的类对象(千万不要指定给创建的对象指定父对象[this]);
 * ----5. 将MyWork对象移动到创建的子线程对象中, 需要调用QObject类提供的moveToThread()方法;
 * ----6. 启动子线程,调用 start(),这时候线程启动了,但是移动到线程中的对象并没有工作;
 * ----7. 调用MyWork类对象的工作函数,让这个函数开始执行,这时候是在移动到的那个子线程中运行的;
 * 
 * 使用这种多线程方式,假设有多个不相关的业务流程需要被处理,那么就可以创建多个类似于MyWork的类,
 * 将业务流程放多类的公共成员函数中, 然后将这个业务类的实例对象移动到对应的子线程中moveToThread()
 * 就可以了, 这样可以让编写的程序更加灵活, 可读性更强，更易于维护.
 * 
 */

#include "src/Task_test.h"
#include "src/Thread_test.h"

#include <QApplication>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    ThreadTest test;
    test.show();

    TaskTest task;
    task.show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
