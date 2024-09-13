/**
 * @file 09_event_processor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt是一个基于C++的框架,主要用来开发带窗口的应用程序(不带窗口的也行,但不是主流),
 * 使用的基于窗口的应用程序都是基于事件, 其目的主要是用来实现回调(因为只有这样程序的效率才是最高的),
 * 所以在Qt框架内部提供了一些列的事件处理机制, 当窗口事件产生之后,
 * *事件会经过：事件派发 -> 事件过滤->事件分发->事件处理几个阶段.
 * ?Qt窗口中对于产生的一系列事件都有默认的处理动作,如果有特殊需求就需要在合适的阶段重写事件的处理动作;
 *
 * 事件(event)是由系统或者Qt本身在不同的场景下发出的; 
 * 当用户按下/移动鼠标、敲下键盘，或者是窗口关闭/大小发生变化/隐藏或显示都会发出一个相应的事件;
 * 一些事件在对用户操作做出响应时发出，如鼠标/键盘事件等;
 * 另一些事件则是由系统自动发出，如计时器事件;
 * 
 * *每一个Qt应用程序都对应一个唯一的 QApplication应用程序对象,
 * 然后调用这个对象的exec()函数, 这样Qt框架内部的事件检测就开始了
 * ?(程序将进入事件循环来监听应用程序的事件);
1. 当事件产生之后，Qt使用用应用程序对象调用notify()函数将事件发送到指定的窗口;
[override virtual] bool QApplication::notify(QObject *receiver, QEvent *e);
2. 事件在发送过程中可以通过事件过滤器进行过滤，默认不对任何产生的事件进行过滤;
// 需要先给窗口安装过滤器, 该事件才会触发
[virtual] bool QObject::eventFilter(QObject *watched, QEvent *event)
3. 当事件发送到指定窗口之后，窗口的事件分发器会对收到的事件进行分类;
[override virtual protected] bool QWidget::event(QEvent *event);
4. 事件分发器会将分类之后的事件(鼠标事件、键盘事件、绘图事件...)分发给对应的事件处理器函数进行处理,
   每个事件处理器函数都有默认的处理动作(可以重写这些事件处理器函数), 比如: 鼠标事件.
// 鼠标按下
[virtual protected] void QWidget::mousePressEvent(QMouseEvent *event);
// 鼠标释放
[virtual protected] void QWidget::mouseReleaseEvent(QMouseEvent *event);
// 鼠标移动
[virtual protected] void QWidget::mouseMoveEvent(QMouseEvent *event);
// -------------------------------------------------------------------
 * 
 * ?2. 事件处理器函数
 * Qt的事件处理器函数处于食物链的最末端,每个事件处理器函数都对应一个唯一的事件,
 * 这为重新定义事件的处理动作提供了便利. 另外Qt提供的这些事件处理器函数都是回调函数,
 * *也就是说作为使用者只需要指定函数的处理动作,关于函数的调用是不需要操心的,当某个事件被触发,Qt框架会调用对应的事件处理器函数;
 * QWidget类是Qt中所有窗口类的基类,在这个类里边定义了很多事件处理器函数,它们都是受保护的虚函数;
 * ?可以在Qt的任意一个窗口类中重写这些虚函数来重定义它们的行为.
 * 
 * 2.1 鼠标事件:按下,释放,移动,双击,进入,离开;
 * 2.2 键盘事件:按下,释放;
 * 2.3 窗口重绘事件:当窗口需要刷新的时候，该函数就会自动被调用;
[virtual protected] void QWidget::paintEvent(QPaintEvent *event);
 * 2.4 窗口关闭事件:当窗口标题栏的关闭按钮被按下并且在窗口关闭之前该函数被调用;
[virtual protected] void QWidget::closeEvent(QCloseEvent *event);
 * 2.5 重置窗口大小事件:当窗口的大小发生变化，该函数被调用;
 * *Qt的帮助文档，窗口的事件处理器函数:
1. 受保护的虚函数;
2. 函数名分为两部分: 事件描述+Event;
3. 函数带一个事件类型的参数;
 * 
 * 3. 重写事件处理器函数
 * 由于事件处理器函数都是虚函数,添加一个标准窗口类的派生类,这样不仅使子类继承了父类的属性,
 * 还可以在这个子类中重写父类的虚函数.
 * 
 * 
 * 
 */
#include "src/mainwindow_event.h" // 生成的窗口类头文件

#include <QApplication>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    // 创建窗口类对象
    MainWindow window;
    // 显示窗口
    window.show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
