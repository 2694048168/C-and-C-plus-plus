/**
 * @file 05_layout_widget.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt的窗口布局以及使用, 主要内容包括: 布局的样式, 基于UI界面设置布局, 基于API设置布局;
 * ====1. 布局的样式
 * *Qt窗口布局是指将多个子窗口按照某种排列方式将其全部展示到对应的父窗口中的一种处理方式
 * ?水平布局	   布局中的所有的控件水平排列	    1行	       N列（N>=1）
 * ?垂直布局	   布局中的所有的控件垂直排列	    N行(N>=1） 	1列
 * ?网格(栅格)布局	布局中的所有的控件垂直+水平排列	 N行	    N列 (N>=1）
 *  
 * ====2. 在UI窗口中设置布局
 * 第一种方式是使用Qt提供的布局, 从工具箱中找到相关的布局, 然后将其拖拽到UI窗口中,
 * ----将相应的控件放入到布局对应的红色框内部, 这些控件就按照布局的样式自动排列到一起.
 * 第二种方式是直接在父窗口中对子部件进行布局, 如果窗口结构很复杂需要嵌套, 
 * 那么就需要先将这些子部件放到一个容器类型的窗口中, 然后再对这个容器类型的窗口进行布局操作.
 * ?进行窗口布局的时候为了让界面看起来更加美观, 需要调整控件的位置, 比如: 靠左, 靠右, 居中, 
 * 又或者需要调节两个控件之间的距离, 以上这些需求使用弹簧都是可以实现的.
 * ?布局属性设置;
 * ?布局的注意事项: 如果某个窗口没有进行布局, 那么当这个窗口显示出来之后里边的子部件就可能无法被显示出来.
 * 
 */

#include <QApplication>
#include <QGridLayout>
#include <QPushButton>
#include <QTextEdit>
#include <QWidget>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    // 创建父窗口对象
    QWidget     *window = new QWidget;
    // 创建子窗口对象
    QPushButton *button1 = new QPushButton("One");
    QPushButton *button2 = new QPushButton("Two");
    QPushButton *button3 = new QPushButton("Three");
    QPushButton *button4 = new QPushButton("Four");
    QPushButton *button5 = new QPushButton("Five");
    QPushButton *button6 = new QPushButton("Six");
    // 多行文本编辑框
    QTextEdit   *txedit = new QTextEdit;
    txedit->setText("我占用了两行两列的空间哦。");

    QGridLayout *layout = new QGridLayout;
    // 按钮起始位置: 第1行, 第1列, 该按钮占用空间情况为1行1列
    layout->addWidget(button1, 0, 0);
    // 按钮起始位置: 第1行, 第2列, 该按钮占用空间情况为1行1列
    layout->addWidget(button2, 0, 1);
    // 按钮起始位置: 第1行, 第3列, 该按钮占用空间情况为1行1列
    layout->addWidget(button3, 0, 2);
    // 编辑框起始位置: 第2行, 第1列, 该按钮占用空间情况为2行2列
    layout->addWidget(txedit, 1, 0, 2, 2);
    // 按钮起始位置: 第2行, 第3列, 该按钮占用空间情况为1行1列
    layout->addWidget(button4, 1, 2);
    // 按钮起始位置: 第3行, 第3列, 该按钮占用空间情况为1行1列
    layout->addWidget(button5, 2, 2);
    // 按钮起始位置: 第4行, 第1列, 该按钮占用空间情况为1行3列
    layout->addWidget(button6, 3, 0, 1, 3);

    // 网格布局设置给父窗口对象
    window->setLayout(layout);
    // 显示父窗口
    window->show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
