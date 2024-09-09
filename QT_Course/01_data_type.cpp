/**
 * @file 01_data_type.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt中常用的数据类型,主要内容包括: 基础数据类型, Log日志输出, 字符串类型, 
 * *QVariant, 位置和尺寸相关类型, 日期和时间相关类型.
 * 
 * ====1. 基础类型
 * 因为Qt是一个C++ 框架, 因此C++中所有的语法和数据类型在Qt中都是被支持的, 
 * ?但是Qt中也定义了一些属于自己的数据类型, QT基本数据类型定义在
 *  #include <QtGlobal> 中, QT基本数据类型有:
 * --------------------------------------
 * qint8	signed char	    有符号8位数据
 * uchar	unsigned char	无符号字符类型
 * uint	    unsigned int	无符号整型
 * --------------------------------------
 * 
 * ====2. log输出
 * 在Qt中进行log输出, 一般不使用c中的printf, 也不是使用C++中的cout, 
 * *Qt框架提供了专门用于日志输出的类, 头文件名为 QDebug;
 * ?默认情况下日志信息是不会打印到终端窗口的, 如果想要实现这样的效果, 必须在项目文件中添加相关的属性信息;
 * 设置IDE的配置项 config, 添加 console 控制台属性(VS2022);
 * 
 * =====3. 字符串类型
 * ----------------------------------------
 * C    | char*
 * C++  | const char*, std::string, char*
 * QT   | QByteArray, QString
 * ----------------------------------------
 * Qt中QByteArray可以看做是c语言中 char*的升级版本,
 * 在使用这种类型的时候可通过这个类的构造函数申请一块动态内存,用于存储需要处理的字符串(字符数组)数据;
 * ?常用的一些API函数, 养成遇到问题主动查询帮助文档的好习惯.
 * 类型转换
// ================================================
// 将QByteArray类型的字符串 转换为 char* 类型
char *QByteArray::data();
const char *QByteArray::data() const;

// int, short, long, float, double -> QByteArray
// 其他重载的同名函数可参考Qt帮助文档, 此处略
QByteArray &QByteArray::setNum(int n, int base = 10);
QByteArray &QByteArray::setNum(short n, int base = 10);
QByteArray &QByteArray::setNum(qlonglong n, int base = 10);
QByteArray &QByteArray::setNum(float n, char f = 'g', int prec = 6);
QByteArray &QByteArray::setNum(double n, char f = 'g', int prec = 6);
[static] QByteArray QByteArray::number(int n, int base = 10);
[static] QByteArray QByteArray::number(qlonglong n, int base = 10);
[static] QByteArray QByteArray::number(double n, char f = 'g', int prec = 6);

// QByteArray -> int, short, long, float, double
int QByteArray::toInt(bool *ok = Q_NULLPTR, int base = 10) const;
short QByteArray::toShort(bool *ok = Q_NULLPTR, int base = 10) const;
long QByteArray::toLong(bool *ok = Q_NULLPTR, int base = 10) const;
float QByteArray::toFloat(bool *ok = Q_NULLPTR) const;
double QByteArray::toDouble(bool *ok = Q_NULLPTR) const;

// std::string -> QByteArray
[static] QByteArray QByteArray::fromStdString(const std::string &str);
// QByteArray -> std::string
std::string QByteArray::toStdString() const;

// 所有字符转换为大写
QByteArray QByteArray::toUpper() const;
// 所有字符转换为小写
QByteArray QByteArray::toLower() const;
// ================================================
 * QString也是封装了字符串,内部的编码为utf8, 
 * UTF-8属于Unicode字符集, 它固定使用多个字节(window为2字节, linux为3字节)来表示一个字符.
 * ?常用的一些API函数, 养成遇到问题主动查询帮助文档的好习惯.
 * 类型转换
// ================================================
// 将int, short, long, float, double 转换为 QString 类型
// 其他重载的同名函数可参考Qt帮助文档, 此处略
QString &QString::setNum(int n, int base = 10);
QString &QString::setNum(short n, int base = 10);
QString &QString::setNum(long n, int base = 10);
QString &QString::setNum(float n, char format = 'g', int precision = 6);
QString &QString::setNum(double n, char format = 'g', int precision = 6);
[static] QString QString::number(long n, int base = 10);
[static] QString QString::number(int n, int base = 10);
[static] QString QString::number(double n, char format = 'g', int precision = 6);

// 将 QString 转换为 int, short, long, float, double 类型
int QString::toInt(bool *ok = Q_NULLPTR, int base = 10) const;
short QString::toShort(bool *ok = Q_NULLPTR, int base = 10) const;
long QString::toLong(bool *ok = Q_NULLPTR, int base = 10) const
float QString::toFloat(bool *ok = Q_NULLPTR) const;
double QString::toDouble(bool *ok = Q_NULLPTR) const;

// 将标准C++中的 std::string 类型 转换为 QString 类型
[static] QString QString::fromStdString(const std::string &str);
// 将 QString 转换为 标准C++中的 std::string 类型
std::string QString::toStdString() const;

// QString -> QByteArray
// 转换为本地编码, 跟随操作系统
QByteArray QString::toLocal8Bit() const;
// 转换为 Latin-1 编码的字符串 不支持中文
QByteArray QString::toLatin1() const;
// 转换为 utf8 编码格式的字符串 (常用)
QByteArray QString::toUtf8() const;

// 所有字符转换为大写
QString QString::toUpper() const;
// 所有字符转换为小写
QString QString::toLower() const;
// ================================================
 * =====4. QVariant
 * QVariant这个类很神奇,或者说方便;很多时候，需要几种不同的数据类型需要传递,
 * 如果用结构体，又不大方便,容器保存的也只是一种数据类型,而QVariant则可以统统搞定.
 * *QVariant 这个类型充当着最常见的数据类型的联合. 
 * QVariant 可以保存很多Qt的数据类型, 包括QBrush、QColor、QCursor、QDateTime、QFont、
 * QKeySequence、 QPalette、QPen、QPixmap、QPoint、QRect、QRegion、QSize和QString,
 * 并且还有C++基本类型，如 int、float等.
 * 
 * =====5. 位置和尺寸
 * *在QT中常见的 点, 线, 尺寸, 矩形 都被进行了封装;
 * 1. QPoint类封装常用的坐标点 (x, y);
 * 2. QLine是一个直线类,封装了两个坐标点(两点确定一条直线);
 * 3. 在QT中QSize类用来形容长度和宽度;
 * 4. 在Qt中使用 QRect类来描述一个矩形;
 * 
 * =====6. 日期和时间
 * *QDate类可以封装日期信息也可以通过这个类得到日期相关的信息, 包括:年, 月, 日;
 * *QTime类可以封装时间信息也可以通过这个类得到时间相关的信息, 包括:时, 分, 秒, 毫秒;
 * *QDateTime类可以封装日期和时间信息也可以通过这个类得到日期和时间相关的信息, 包括:年, 月, 日, 时, 分, 秒, 毫秒;
 * 
 * 
 * 
 * 
 */

#include <QApplication> // 应用程序类头文件
#include <QDate>
#include <QDebug>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    // 包含了QDebug头文件, 直接通过全局函数 qDebug() 就可以进行日志输出了
    auto coordinate = QRect(0, 10, 50, 40);
    qDebug() << "Date:" << QDate::currentDate();
    qDebug() << "Types:" << QString("String") << QChar('x') << coordinate;
    qDebug() << "Custom coordinate type:" << coordinate;

    // 和全局函数 qDebug() 类似的日志函数还有: qWarning(), qInfo(), qCritical()
    int   number = 666;
    float i      = 11.11;
    qWarning() << "Number:" << number << "Other value:" << i;
    qInfo() << "Number:" << number << "Other value:" << i;
    qCritical() << "Number:" << number << "Other value:" << i;

    qDebug() << "我是要成为海贼王的男人!!!";
    qDebug() << "我是隔壁的二柱子...";
    qDebug() << "我是鸣人, 我擅长嘴遁!!!\n";

    // ========= QString 字符串格式 =========
    // 其他重载的同名函数可参考Qt帮助文档, 此处略
    // QString QString::arg(const QString &a, int fieldWidth = 0, QChar fillChar = QLatin1Char( ' ' )) const;
    // QString QString::arg(int a, int fieldWidth = 0, int base = 10, QChar fillChar = QLatin1Char( ' ' )) const;
    int     idx      = 12;              // 假设该变量表示当前文件的编号
    int     total    = 42;              // 假设该变量表示文件的总个数
    QString fileName = "filename.json"; // 假设该变量表示当前文件的名字
    // 使用以上三个变量拼接一个动态字符串
    QString status = QString("Processing file %1 of %2: %3").arg(idx).arg(total).arg(fileName);
    std::cout << status.toStdString() << std::endl;

    // ========= QVariant =========
#if 1
    QVariant v;
    v.setValue(5);
#else
    QVariant v = QVariant::fromValue(5);
#endif

    int     num = v.toInt();    // i is now 5
    QString str = v.toString(); // s is now "5"
    std::cout << num << std::endl;
    std::cout << str.toStdString() << std::endl;
    // 该函数的返回值是一个枚举类型, 可通过这个枚举判断出实际是什么类型的数据
    // Type QVariant::type() const;
    // ?返回值Type的枚举定义, 全部信息可以自行查阅Qt帮助文档

    // ========= date and time =========
    // 静态函数 得到当前时区的日期和时间(本地设置的时区对应的日期和时间)
    // [static] QDateTime QDateTime::currentDateTime();
    QDateTime current_datetime = QDateTime::currentDateTime();
    std::cout << current_datetime.toString("yyyy-MM-dd HH:mm:ss ap").toStdString() << std::endl;

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
