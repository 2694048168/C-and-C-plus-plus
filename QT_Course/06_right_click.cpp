/**
 * @file 06_right_click.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 如果想要在某一窗口中显示右键菜单, 其处理方式大体上有两种, 
 * *这两种方式分别为基于鼠标事件实现和基于窗口的菜单策略实现.
 * ?其中第二种方式中又有三种不同的实现方式, 因此如果想要在窗口中显示一个右键菜单一共四种实现方式.
 * 
 * ====1. 基于鼠标事件实现
 * 使用这种方式实现右键菜单的显示需要使用事件处理器函数, 在Qt中这类函数都是回调函数, 
 * 并且在自定义窗口类中我们还可以自定义事件处理器函数的行为
 *（因为子类继承了父类的这个方法并且这类函数是虚函数）.
 * 
 * ====2. 基于窗口的菜单策略实现
 * 这种方式是使用 Qt 中 QWidget类中的右键菜单函数 
 * *QWidget::setContextMenuPolicy(Qt::ContextMenuPolicy policy) 来实现, 
 * 因为这个函数的参数可以指定不同的值, 因此不同参数对应的具体的实现方式也不同.
void QWidget::setContextMenuPolicy(Qt::ContextMenuPolicy policy);
参数: 	
  - Qt::NoContextMenu	     --> 不能实现右键菜单
  - Qt::PreventContextMenu   --> 不能实现右键菜单
  - Qt::DefaultContextMenu   --> 基于事件处理器函数 QWidget::contextMenuEvent() 实现
  - Qt::ActionsContextMenu   --> 添加到当前窗口中所有 QAction 都会作为右键菜单项显示出来
  - Qt::CustomContextMenu    --> 基于 QWidget::customContextMenuRequested() 信号实现
// ---------------------------------------------------------------
 * ? https://subingwen.cn/qt/qt-right-menu/#2-1-Qt-DefaultContextMenu
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
