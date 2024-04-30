/**
 * @file 02_interfaceInjection.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 更新银行日志记录器
 * Logger 接口允许提供多个记录器实现, 
 * 这允许Logger消费者使用 log_transfer 方法记录传输的日志, 而不必知道记录器的实现细节.
 * *看如何使用构造函数注入或属性注入来更新 Bank
 * 
 * ==== Constructor Injection 构造函数注入
 * 使用构造函数注入, 可以将 Logger 引用传递到 Bank 类的构造函数中.
 * 通过这种方式, 可以建立特定的 Bank 实例化将使用的日志记录方式.
 * Bank 类的构造函数使用成员初始化列表来设置 logger 的值,
 * 因为引用不能被重定位, 所以在Bank 的生命周期内, logger 指向的对象不会改变,
 * 这相当于在 Bank 构造时固定了 logger 选择.
 * 
 * ==== Property Injection 属性注入
 * 也可以使用属性注入而非构造函数注入将 logger 插入 Bank,
 * 这种方法使用指针而不是引用, 因为指针可以被重定位(不像引用),
 * 所以可以随时改变 Bank 的行为.
 * set_logger 方法能够让在生命周期的任何时候将新的日志记录器注入 Bank 对象,
 * 当将日志记录器设置为 ConsoleLogger 实例时, 日志输出中会有一个 [cons]前缀,
 * 当将日志记录器设置为 FileLogger 实例时, 会得到一个[file]前缀.
 * 
 */
class Logger
{
public:
    virtual ~Logger() = default;

    virtual void log_transfer(long from, long to, double amount) = 0;
};

class ConsoleLogger : public Logger
{
public:
    void log_transfer(long from, long to, double amount) override
    {
        printf("[cons] %ld -> %ld: %f\n", from, to, amount);
    }
};

class FileLogger : public Logger
{
public:
    void log_transfer(long from, long to, double amount) override
    {
        printf("[file] %ld,%ld,%f\n", from, to, amount);
    }
};

// Constructor Injection
class BankConstructorInjection
{
public:
    BankConstructorInjection(Logger &logger)
        : m_logger{logger}
    {
    }

    void make_transfer(long from, long to, double amount)
    {
        //  --snip--
        m_logger.log_transfer(from, to, amount);
    }

private:
    Logger &m_logger;
};

// Property Injection
struct BankPropertyInjection
{
    void set_logger(Logger *new_logger)
    {
        m_logger = new_logger;
    }

    void make_transfer(long from, long to, double amount)
    {
        if (m_logger)
            m_logger->log_transfer(from, to, amount);
    }

private:
    Logger *m_logger{};
};

/** Choosing Constructor or Property Injection 
 * @brief 构造函数注入和属性注入的选择
 * 选择构造函数注入还是属性注入取决于设计要求,
 * 如果需要在对象的整个生命周期中修改对象成员的底层类型, 那么应该选择指针和属性注入方法;
 * 但是灵活使用指针和属性注入是有代价的. 必须确保要么不将 logger 设置为 nullptr, 
 * 要么在使用 logger 之前检查这个条件, 还有一个问题是默认行为是什么 logger的初始值是多少?
 * 
 * *还可以同时使用构造函数注入和属性注入, 这鼓励使用类的人去考虑初始化
 */
class Bank
{
public:
    // Constructor Injection
    Bank(Logger *logger)
        : m_logger{logger} {};

    void set_logger(Logger *new_logger)
    {
        m_logger = new_logger;
    }

    void make_transfer(long from, long to, double amount)
    {
        if (m_logger)
            m_logger->log_transfer(from, to, amount);
    }

private:
    // Property Injection
    Logger *m_logger;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    ConsoleLogger            logger;
    BankConstructorInjection bank{logger};
    bank.make_transfer(1000, 2000, 49.95);
    bank.make_transfer(2000, 4000, 20.00);

    printf("===============================\n");
    ConsoleLogger console_logger;
    FileLogger    file_logger;

    BankPropertyInjection bank_;
    bank_.set_logger(&console_logger);
    bank_.make_transfer(1000, 2000, 49.95);
    bank_.set_logger(&file_logger);
    bank_.make_transfer(2000, 4000, 20.00);

    /**
     * @brief 介绍了接口的定义方法, 虚函数在继承中的核心作用,
     * 以及使用构造函数注入和属性注入的一般规则,
     * 无论选择哪种方法, 接口继承和对象组合结合起来都可为大多数运行时多态应用程序提供足够的灵活性;
     * 可以用很少的开销甚至不需要开销来实现类型安全的运行时多态;
     * 接口鼓励封装和松耦合的设计,
     * 通过简单, 集中的接口, 可以使代码在不同的项目中可移植, 从而鼓励代码重用.
     * 
     */

    return 0;
}
