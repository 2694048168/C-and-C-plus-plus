/**
 * @file 16_database_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt数据库操作
 * Qt框架中对数据库操作提供了很好的支持,通过Qt提供的类非常方便地和本地或者远程数据库进行连接.
 * ?数据库是 C-S（client-server）结构的, 要连接的数据库属于服务器端,通过Qt编写的应用程序属于客户端.
 * 
 * Qt提供的常用的数据库类:
 * *1. QSqlDatabase: 通过这个类添加/删除/复制/关闭数据库实例;
 * *2. QSqlQuery: 数据库查询类;
 * *3. QSqlRecord: 数据库记录（通常是数据库中表或视图中的一行）的功能和特征;
 * *4. QSqlField: 数据库表或视图中单个列(字段)的特征, 例如数据类型和列名;
 * *5. QSqlQueryModel: 执行SQL语句和遍历结果集的高级接口,它构建在底层QSqlQuery之上,可以用来为视图类(如QTableView)提供数据;
 * *6. QSqlError: 数据操作失败可以通过这个类获取相关的错误信息;
 * 
 * ?使用者能够自己通过SQL语句实现数据库表数据的添、删、查、改.
 * 
 * Qt中不论连接的何种类型的关系型数据库,其操作流程是一致的:
 * 1. 创建数据库实例并初始化;
 * 2. 连接数据库;
 * 3. 对数据库进行一系列的添、删、查、改操作（编写并执行SQL语句）;
 * 4. 关闭数据库;
 * 
 * =====2.1 QSqlDatabase
 * QSqlDatabase类提供了一个连接访问数据库的接口, 一个QSqlDatabase的实例就表示一个连接.
 * *该连接通过受支持的数据库驱动程序之一提供对数据库的访问, 这些驱动程序派生自QSqlDriver.
 * !在高版本的Qt中,有些数据库的驱动/插件(本质是动态库)需要自己编译,比如：MySQL、Oracle.
 * !如果没有对应的驱动/插件是无法进行数据库连接的;
 * 
[static] QSqlDatabase QSqlDatabase::addDatabase(
    const QString &type, 
    const QString &connectionName = QLatin1String( defaultConnection ));
该函数的有两个参数分别是：
1. type: 指定要连接什么样的数据库, 就是数据库驱动对应的驱动名
2. connectionName: 数据库连接名, 默认叫: defaultConnection. 
可以在应用程序中添加多个数据库连接（也就是多个实例），每个连接都对应一个唯一的名字.
函数的返回值就是得到的数据库实例对象。
// ==================================================
 * 返回有关数据库上发生的最后一个错误的信息
QSqlError QSqlDatabase::lastError() const;
QString QSqlError::text() const;
// ==================================================
 * 
 * ? https://subingwen.cn/qt/qt-db/
 * 
 */

#include <QApplication>
#include <QDebug>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <iostream>

void initDatabase()
{
    //qDebug()<<QSqlDatabase::drivers();//打印驱动列表
    QSqlDatabase database;
    //检测已连接的方式 - 默认连接名
    //QSqlDatabase::contains(QSqlDatabase::defaultConnection)
    if (QSqlDatabase::contains("qt_sql_default_connection"))
        database = QSqlDatabase::database("qt_sql_default_connection");
    else
        database = QSqlDatabase::addDatabase("QSQLITE"); // sqlite

    //检测已连接的方式 - 自定义连接名
    /*if(QSqlDatabase::contains("mysql_connection"))
        database = QSqlDatabase::database("mysql_connection");
    else
        database = QSqlDatabase::addDatabase("QSQLITE","mysql_connection");*/

    //设置数据库路径，不存在则创建
    database.setDatabaseName("sqlite.db");
    //database.setUserName("Ithaca");  //SQLite不需要用户名和密码
    //database.setPassword("root");

    // 如果需要在内存中创建数据库，而不是指定一个文件，可以setDatabaseName(":memory:");
    // database.setDatabaseName(":memory:");

    //打开数据库
    if (database.open())
    {
        qDebug() << "open database successfully\n";
        //关闭数据库
        database.close();
    }
}

void createTable()
{
    //sql语句不熟悉的推荐《sql必知必会》,轻松入门
    //如果不存在则创建my_table表
    //id自增,name唯一
    const QString sql = R"(
                      CREATE TABLE IF NOT EXISTS  my_table (
                      id   INTEGER   PRIMARY KEY AUTOINCREMENT NOT NULL,
                      name CHAR (50) UNIQUE NOT NULL,
                      age  INTEGER
                      );)";
    //QSqlQuery构造前，需要db已打开并连接
    //未指定db或者db无效时使用默认连接进行操作
    QSqlQuery     query;

    if (query.exec(sql))
    {
        qDebug() << "init table successfully\n";
    }
    else
    {
        //打印sql语句错误信息
        qDebug() << "init table error: " << query.lastError();
    }
}

//插入数据
void insertRecord(const QString &name, int age)
{
    QSqlQuery query;
    //方式一，直接执行SQL语句
    query.exec(QString(R"(INSERT INTO my_table(name,age) VALUES('%1',%2);)").arg(name).arg(age));
    //方式二，绑定值，待定变量默认用问号占位,注意字符串也没有引号
    /*query.prepare(R"(INSERT INTO my_table(name,age) VALUES(?,?);)");
    query.addBindValue(name);
    query.addBindValue(age);
    query.exec();*/
}

//删除数据
void deleteRecord(const QString &name)
{
    QSqlQuery query;
    //方式一，直接执行SQL语句
    query.exec(QString(R"(DELETE FROM my_table WHERE name='%1';)").arg(name));
    //方式二，绑定值，待定变量默认用问号占位
    /*query.prepare(R"(DELETE FROM my_table WHERE name=?;)");
    query.addBindValue(name);
    query.exec();*/
}

//更新数据
void updateRecord(const QString &name, int age)
{
    QSqlQuery query;
    //方式一，直接执行SQL语句
    query.exec(QString(R"(UPDATE my_table SET age=%2 WHERE name='%1';)").arg(name).arg(age));
    //方式二，绑定值，待定变量默认问号，可自定义
    /*query.prepare(R"(UPDATE my_table SET age=:age WHERE name=:name;)");
    query.bindValue(":name",name);//通过自定义的别名来替代
    query.bindValue(":age",age);
    query.exec();*/
}

//查询数据
int searchRecord(const QString &name)
{
    QSqlQuery query;
    query.exec(QString(R"(SELECT age FROM my_table WHERE name='%1';)").arg(name));

    //获取查询结果的第0个值，
    //如果结果是多行数据，可用while(query.next()){}遍历每一行
    int ageValue = -1;
    if (query.next())
    {
        ageValue = query.value(0).toInt();
    }
    qDebug() << ageValue;
    return ageValue;
}

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";
    // QMYSQL 是MySQL数据库对应的驱动名
    // QOCI 是Oracle数据库对应的驱动名
    QStringList list = QSqlDatabase::drivers();
    qDebug() << list;

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
