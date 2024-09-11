/**
 * @file 02_signal_slot.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt中的信号槽,主要内容包括: 信号槽的本质, 信号槽的关系, 标准信号槽的使用, 
 * *自定义信号槽的使用, 信号槽的拓展, Lambda表达式.
 * 
 * ====1. 信号和槽概述
 * 信号槽是 Qt 框架引以为豪的机制之一,所谓信号槽,实际就是观察者模式(发布-订阅模式).
 * 当某个事件发生之后，比如,按钮检测到自己被点击了一下,它就会发出一个信号（signal）,
 * 这种发出是没有目的的, 类似广播; 如果有对象对这个信号感兴趣，它就会使用连接（connect）函数,
 * 将想要处理的信号和自己的一个函数（称为槽（slot））绑定来处理这个信号.
 * 也就是说，当信号发出时，被连接的槽函数会自动被回调, 这就类似观察者模式: 
 * 当发生了感兴趣的事件，某一个操作就会被自动触发.
 * *信号的本质就是事件,在QT中信号的发出者是某个实例化的类对象,对象内部可以进行相关事件的检测;
 * *槽函数是一类特殊的功能的函数(Qt中槽函数的所有者也是某个类的实例对象), 可以作为类的普通成员函数来使用,对Qt框架中产生的信号进行处理;
 * *在Qt中信号和槽函数都是独立的个体,本身没有任何联系,但是由于某种特性需求可以将二者连接到一起,使用QOjbect类中的connect函数进二者的关联;
// 使用connect()进行信号槽连接的注意事项:
1. connect函数相对于做了信号处理动作的注册;
2. 调用concent函数的sender对象的信号并没有产生, 因此 receiver 对象的method也不会被调用;
3. method槽函数本质是一个回调函数, 调用的时机是信号产生之后, 调用是Qt框架来执行的;
4. connect中的sender和receiver两个指针必须被实例化了, 否则concent不会成功;
 * ?标准信号槽使用, connect()操作一般写在窗口的构造函数中, 相当于在事件产生之前在qt框架中先进行注册, 
 * 这样在程序运行过程中假设产生了按钮的点击事件, 框架就会调用信号接收者对象对应的槽函数了;
 * 如果信号不产生, 槽函数也就一直不会被调用.
 * 
 * ====3. 自定义信号槽使用
 * 如果想要在QT类中自定义信号槽, 需要满足一些条件, 并且有些事项也需要注意:
 * 1. 要编写新的类并且让其继承Qt的某些标准类;
 * 2. 这个新的子类必须从QObject类或者是QObject子类进行派生;
 * 3. 在定义类的头文件中加入 Q_OBJECT 宏;
 * ?自定义信号
 * 1. 信号是类的成员函数;
 * 2. 返回值必须是 void 类型;
 * 3. 信号的名字可以根据实际情况进行指定;
 * 4. 参数可以随意指定, 信号也支持重载;
 * 5. 信号需要使用 signals 关键字进行声明, 使用方法类似于public等关键字;
 * 6. 信号函数只需要声明, 不需要定义(没有函数体实现);
 * 7. 在程序中发射自定义信号: 发送信号的本质就是调用信号函数;
 * ------习惯性在信号函数前加关键字: emit, 但是可以省略不写;
 * ------emit只是显示的声明一下信号要被发射了, 没有特殊含义;
 * ------底层 emit == #define emit
// ==================================================
qRegisterMetaType使用方法如下：
1、注册位置：在第一次使用此类链接跨线程的signal/slot之前，一般在当前类的构造函数中进行注册；
2、注册方法：在当前类的顶部包含：#include <QMetaType>，构造函数中加入代码：qRegisterMetaType<custom-class>("custom-class")；
3、custom-class的引用类型需单独注册：qRegisterMetaType<custom-class>("custom-class&")；
4. 若对象包含在自定义的命名空间中时，注册时要带上完整的命令空间: qRegisterMetaType<NSP::custom-class>("NSP::custom-class&")；
5. 若要用信号槽传递其智能指针，也需要注册:qRegisterMetaType<QSharedPointer<MyStruct>>("QSharedPointer<MyStruct>");
// ==================================================
 * ?自定义槽
 * 1. 返回值必须是 void 类型;
 * 2. 槽也是函数, 因此也支持重载;
 * 3. 槽函数需要指定多少个参数, 需要看连接的信号的参数个数;
 * 4. 槽函数的参数是用来接收信号传递的数据的, 信号传递的数据就是信号的参数
// ==============================
举例:
信号函数: void sg_test(int a, double b);
槽函数: void sl_test(int a, double b);
总结:
槽函数的参数应该和对应的信号的参数个数, 从左到右类型依次对应
信号的参数可以大于等于槽函数的参数个数 == 信号传递的数据被忽略了
信号函数: void sg_test(int a, double b);
槽函数: void sl_test(int a);
// ==============================
 * 5. Qt中槽函数的类型是多样的: 类的成员函数、全局函数、静态函数、Lambda表达式（匿名函数）;
 * 6. 槽函数可以使用关键字进行声明: slots (Qt5中slots可以省略不写);
 * ?信号槽拓展
 * 1. 一个信号可以连接多个槽函数, 发送一个信号有多个处理动作
 * ------需要写多个connect（）连接;
 * ------槽函数的执行顺序是随机的, 和connect函数的调用顺序没有关系;
 * ------信号的接收者可以是一个对象, 也可以是多个对象;
 * 2. 一个槽函数可以连接多个信号, 多个不同的信号, 处理动作是相同的;
 * ------需要写多个connect（）连接;
 * 3. 信号可以连接信号
 * ------信号接收者可以不处理接收的信号, 而是继续发射新的信号，这相当于传递了数据, 并没有对数据进行处理
 * 4. 信号槽是可以断开的 
 * ?信号槽的连接方式
QMetaObject::Connection QObject::connect(
    	const QObject *sender, PointerToMemberFunction signal, 
        const QObject *receiver, PointerToMemberFunction method, 
 *?		Qt::ConnectionType type = Qt::AutoConnection);
 * ?当信号槽函数被重载之后, Qt4的信号槽连接方式不受影响(SIGNAL & SLOT)
 * ?当信号槽函数被重载之后, Qt5中需要给被重载的信号或者槽定义函数指针(如 QComboBox)
// void	currentIndexChanged(int index)
// void	currentTextChanged(const QString &text)
 * 
 */

#include <QApplication> // 应用程序类头文件
#include <QMetaType>
#include <cstddef>
#include <iostream>
#include <string>

struct Message
{
    size_t      id;
    std::string message;
};

class CustomReceiver : public QObject
{
    Q_OBJECT
public:
    explicit CustomReceiver(QObject *parent = nullptr);

    void recv(Message &msg)
    {
        sl_printMessage(msg);
    }

public slots:

    void sl_printMessage(Message &message)
    {
        std::cout << message.id << " ---> " << message.message << std::endl;
    }

    // void sl_printMessage(Message message);
    void sl_printMessage(std::string &message)
    {
        std::cout << message << std::endl;
    }
};

class CustomSender : public QObject
{
    Q_OBJECT
public:
    explicit CustomSender(QObject *parent = nullptr);

    void send(Message &msg)
    {
        emit sg_hungry(msg);
    }

signals:
    void sg_hungry(Message &message);
    // void sg_hungry(Message message);
    void sg_hungry(std::string &msg);
};

class CustomTest : public QObject
{
    Q_OBJECT
public:
    explicit CustomTest(QObject *parent = nullptr)
    {
        // 定义函数指针指向重载的某一个具体的信号地址
        void (CustomSender::*sg_hungry)(Message &message)         = &CustomSender::sg_hungry;
        void (CustomReceiver::*sl_printMessage)(Message &message) = &CustomReceiver::sl_printMessage;

        // connect(&m_sender, &CustomSender::sg_hungry, &m_receiver, &CustomReceiver::sl_printMessage);
        connect(&m_sender, sg_hungry, &m_receiver, sl_printMessage);
    }

    void send(Message &msg)
    {
        m_sender.send(msg);
    }

    void recv(Message &msg)
    {
        m_receiver.recv(msg);
    }

private:
    CustomSender   m_sender;
    CustomReceiver m_receiver;
};

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    qRegisterMetaType<Message>("Message");
    qRegisterMetaType<Message>("Message&");
    qRegisterMetaType<std::string>("std::string&");
    qRegisterMetaType<std::string>("std::string");

    // CustomTest test;
    // Message    msg{42, "the topic message\n"};
    // test.send(msg);
    // test.recv(msg);

    std::cout << "==========================\n";

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
