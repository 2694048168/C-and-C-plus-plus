/**
 * @file 07_widget_btn.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt中常用的按钮控件, 包括: QAbstractButton, QPushButton, 
 * *QToolButton, QRadioButton, QCheckBox
 * 
 * ====1. 按钮基类 QAbstractButton
1.1 标题和图标
// 参数text的内容显示到按钮上
void QAbstractButton::setText(const QString &text);
// 得到按钮上显示的文本内容, 函数的返回就是
QString QAbstractButton::text() const;

// 得到按钮设置的图标
QIcon icon() const;
// 给按钮设置图标
void setIcon(const QIcon &icon);

// 得到按钮图标大小
QSize iconSize() const
// 设置按钮图标的大小
[slot]void setIconSize(const QSize &size);

1.2 按钮的 Check 属性
// 判断按钮是否设置了checkable属性, 如果设置了点击按钮, 按钮一直处于选中状态
// 默认这个属性是关闭的, not checkable
bool QAbstractButton::isCheckable() const;
// 设置按钮的checkable属性
// 参数为true: 点击按钮, 按钮被选中, 松开鼠标, 按钮不弹起
// 参数为false: 点击按钮, 按钮被选中, 松开鼠标, 按钮弹起
void QAbstractButton::setCheckable(bool);

    
// 判断按钮是不是被按下的选中状态
bool QAbstractButton::isChecked() const;
// 设置按钮的选中状态: true-选中, false-没选中
// 设置该属性前, 必须先进行 checkable属性的设置
void QAbstractButton::setChecked(bool);

1.3 信号 & 槽函数
当按钮被激活时(即，当鼠标光标在按钮内时按下然后释放)，当键入快捷键时，或者当click()或animateClick()被调用时，这个信号被发出。值得注意的是，如果调用setDown()、setChecked()或toggle()，则不会触发此信号。
[signal] void QAbstractButton::clicked(bool checked = false);
// 在按下按钮的时候发射这个信号
[signal] void QAbstractButton::pressed();
// 在释放这个按钮的时候发射直观信号
[signal] void QAbstractButton::released();
// 每当可检查按钮改变其状态时，就会发出此信号。checked在选中按钮时为true，在未选中按钮时为false。
[signal] void QAbstractButton::toggled(bool checked);

// 执行一个动画点击:按钮被立即按下，并在毫秒后释放(默认是100毫秒)。
[slot] void QAbstractButton::animateClick(int msec = 100);
// 执行一次按钮点击, 相当于使用鼠标点击了按钮
[slot] void QAbstractButton::click();

// 参考 1.2 中的函数介绍
[slot] void QAbstractButton::setChecked(bool);
// 设置按钮上图标大小
[slot]void setIconSize(const QSize &size);
// 切换可检查按钮的状态。 checked <==> unchecked
[slot] void QAbstractButton::toggle();
// ------------------------------------------
 * ?QPushButton, 这种类型的按钮是Qt按钮中使用频率最高的普通按钮;
 * ?QToolButton类可以设置弹出的菜单的属性, 以及在显示图标的时候可以设置更多的样式, 增强版的QPushButton;
 * ?QRadioButton是Qt提供的单选按钮, 一般都是以组的方式来使用;
 **在同一窗口中, Qt会认为所有的单选按钮都属于同一组, 如果需要多个单选按钮组, 应该将他们放到不同的子窗口中
 * ?QCheckBox是Qt中的复选框按钮, 可以单独使用, 也可以以组的方式使用(同一组可以同时选中多个);
 * 
 * ? https://subingwen.cn/qt/qt-buttons/
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
    std::cout << "=========== QAbstractButton ===============\n";
    std::cout << "==========================\n";

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
