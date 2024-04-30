/**
 * @file 00_basicPolymorphism.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <stdexcept>

/**
  * @brief Polymorphism  多态
  * 多态代码是指只写一次, 但是可以用不同的类型反复使用的代码.
  * 这种灵活性产生了松耦合且高度可重用的代码, 消除了烦琐的复制和粘贴过程, 使代码更易于维护、更可读.
  * C++提供了两种多态方法: 
  * 1. 一种方法是编译时多态, 包括可以在编译时确定的多态类型;
  * 2. 一种方法是运行时多态, 包含在运行时确定的类型;
  * 具体选择哪种方法取决于是在编译时还是在运行时确定多态代码要使用的类型.
  * 
  * *假设需要实现不同种类的日志记录器,
  * 可能需要远程服务器日志记录器,本地文件日志记录器,甚至是向打印机发送作业的日志记录器.
  * 必须能够在运行时改变程序的日志记录方式
  * (例如, 因为服务器维护, 管理员可能需要从通过网络的日志记录切换到本地文件系统的日志记录).
  * 一个简单的方法是使用 enum class 在各种记录器之间进行切换.
  * 
  */
class ConsoleLogger
{
public:
    void log_transfer(long from, long to, double amount)
    {
        printf("[console] %ld -> %ld: %f\n", from, to, amount);
    }
};

struct FileLogger
{
    void log_transfer(long from, long to, double amount)
    {
        printf("[file] %ld,%ld,%f\n", from, to, amount);
    }
};

enum class LoggerType : int
{
    Console = 0,
    File,

    NUM_TYPE
};

class Bank
{
public:
    void make_transfer(long from, long to, double amount)
    {
        // --snip--
        logger.log_transfer(from, to, amount);
    }

public:
    ConsoleLogger logger;
};

class BankLog
{
public:
    BankLog()
        : m_type{LoggerType::Console}
    {
    }

    void set_logger(LoggerType new_type)
    {
        m_type = new_type;
    }

    void make_transfer(long from, long to, double amount)
    {
        // --snip--
        switch (m_type)
        {
        case LoggerType::Console:
            m_consoleLogger.log_transfer(from, to, amount);
            break;
        case LoggerType::File:
            m_fileLogger.log_transfer(from, to, amount);
            break;
        default:
            throw std::logic_error("Unknown Logger type encountered.");
            break;
        }
    }

private:
    LoggerType    m_type;
    ConsoleLogger m_consoleLogger;
    FileLogger    m_fileLogger;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    Bank bank;
    bank.make_transfer(1000, 2000, 49.95);
    bank.make_transfer(2000, 4000, 20.00);

    /**
     * @brief 添加新的日志记录器
     * 这种方法存在几个设计问题, 增加一种新的日志记录器需要对整个代码进行多处更新:
     * 1. 需要写一个新的记录器类型;
     * 2. 需要在枚举类 LoggerType 中添加新的 enum 值;
     * 3. 必须在 switch 语句6中添加一个新的 case 子句;
     * 4. 必须将新的日志类添加为 Bank 的成员;
     * 
     * 考虑采用另一种方法, 即让 Bank 持有一个指向日志记录器的指针,
     * 这样就可以直接设置指针, 完全摆脱 LoggerType,
     * 而且可以利用日志记录器具有相同的函数原型这一事实, 这就是接口背后的思想
     * 
     */

    BankLog bank_log;
    bank_log.make_transfer(1000, 2000, 49.95);
    bank_log.make_transfer(2000, 4000, 20.00);
    bank_log.set_logger(LoggerType::File);
    bank_log.make_transfer(3000, 2000, 75.00);

    return 0;
}
