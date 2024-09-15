/**
 * @file 15_sql_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt中编译数据库驱动
 * 在项目开发过程中,如果涉及到数据的存储,那么肯定会用到数据库.
 * 常用的关系型数据库(操作的时候需要使用sql语句): mysql、oracle、sql-lite等.
 * ?Qt框架中也同样提供了对这些数据库的支持, 使用Qt提供的数据库类就可以在上层使用相同的接口
 * 对各种数据库进行操作了.
 * ?在Qt底层也对数据库的连接进行了封装,基于不同的连接方式提供了不同的插件
 * （二进制代码，在需要的时候被加载，表现形式为动态库、静态库）
 * *在高版本的Qt中官方并没有提供可用的mysql和oracle插件,只提供了源代码,
 * 需要基于这些源码自行编译得到对应的动态库（静态库）之后,才能被程序加载从而连接上相应的数据库.
 * 
 * !Qt 和数据库的版本:Qt 编译套件的位数和数据库的位数必须相同，都是32位或都是64位;
https://dev.mysql.com/downloads/
// C:\Qt\5.15.2\Src\qtbase\src\plugins\sqldrivers\mysql
 * ?如果想要自己能够编译出 MySql 插件，需要有一个前提条件就是在安装 Qt 的时候安装了相关的源码;
 * 
#include <QDebug>
#include <QMessageBox>
#include <QSqlDatabase>
#include <QSqlError>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QStringList list = QSqlDatabase::drivers();
    qDebug() << list;

    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    if(db.isValid())
    {
        QMessageBox::information(this, "conn", "ok");
    }
    else
    {
        QMessageBox::information(this, "conn", "error");
    }
    db.setHostName("127.0.0.1"); // 本地连接
    db.setPort(3306);       // 如果使用的是默认端口可以不设置
    db.setUserName("root");	// 数据库用户名
    db.setPassword("root"); // 数据库密码
    db.setDatabaseName("mysql"); // 数据库名字
    if(db.open())
    {
        QMessageBox::information(this, "打开数据库", "数据库打开成功, 可以读写数据了......");
    }
    else
    {
        QString msg = "数据库打开失败: " + db.lastError().text();
        QMessageBox::information(this, "打开数据库", msg);
    }
}
 * 
 * ?缺少安全模块的动态库，其实这个库文件就是 openssl 的动态库;
https://dev.mysql.com/downloads/
 * *下载使用C++连接数据库需要的组件（建议下载和本地 mysql 动态库一致的版本）;
 * 
 */

#include <QApplication>
#include <QDebug>
#include <QMessageBox>
#include <QSqlDatabase>
#include <QSqlError>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    QStringList list = QSqlDatabase::drivers();
    qDebug() << list;
    // 默认直接支持数据库: sql-lite, ..., ...
    // QList("QSQLITE", "QODBC", "QPSQL")

    // 其中有一个字符串叫做 QMYSQL 这是连接 MySql 数据库需要的驱动的名字,
    // *如果能够检测到 MySQL 插件就能看到这个字符串,
    // 如果没有按照前面的步骤编译出这个插件放到指定目录,则看不到这个驱动名,自然也无法连接 MySql 数据库;

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
