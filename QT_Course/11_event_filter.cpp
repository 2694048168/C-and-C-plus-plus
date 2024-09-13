/**
 * @file 11_event_filter.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 事件过滤器
 * 除了使用事件分发器来过滤Qt窗口中产生的事件,还可以通过事件过滤器过滤相关的事件.
 * 当Qt的事件通过应用程序对象发送给相关窗口之后,窗口接收到数据之前这个期间可对事件进行过滤,
 * ?过滤掉的事件就不能被继续处理了. QObject有一个eventFilter()函数, 用于建立事件过滤器.
[virtual] bool QObject::eventFilter(QObject *watched, QEvent *event);
参数:
    watched：要过滤的事件的所有者对象
    event：要过滤的具体的事件
返回值：如果想过滤掉这个事件，停止它被进一步处理，返回true，否则返回 false
// =============================================================
既然要过滤传递中的事件，首当其冲还是要搞明白如何通过事件过滤器进行事件的过滤，主要分为两步：
**1. 给要被过滤事件的类对象安装事件过滤器
void QObject::installEventFilter(QObject *filterObj);
假设调用installEventFilter()函数的对象为当前对象，那么就可以基于参数指定的filterObj对象来过滤当前对象中的指定的事件了.
**2. 在要进行事件过滤的类中(filterObj 参数对应的类)重写从QObject类继承的虚函数eventFilter().
 *
ui->textEdit->installEventFilter(窗口A对象);
ui->textEdit->installEventFilter(窗口B对象);
ui->textEdit->installEventFilter(窗口C对象);
 * ?如果一个对象存在多个事件过滤器, 那么最后一个安装的会第一个执行, 也就是说窗口C先进行事件过滤, 然后窗口B, 最后窗口A;
 * ?事件过滤器和被安装过滤器的组件必须在同一线程, 否则过滤器将不起作用;
 * ?另外如果在安装过滤器之后, 这两个组件到了不同的线程, 那么只有等到二者重新回到同一线程的时候过滤器才会有效;
 * 
bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    // 判断对象和事件
    if(watched == ui->textEdit && event->type() == QEvent::KeyPress)
    {
        QKeyEvent* keyEv = (QKeyEvent*)event;
        if(keyEv->key() == Qt::Key_Enter ||         // 小键盘确认
                keyEv->key() == Qt::Key_Return)     // 大键盘回车
        {
            qDebug() << "我是回车, 被按下了...";
            return true;
        }
    }
    return false;
}
 * 
 */

#include <QApplication>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
