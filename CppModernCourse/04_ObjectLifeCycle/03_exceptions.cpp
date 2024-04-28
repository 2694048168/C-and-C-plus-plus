/**
 * @file 03_exceptions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <stdexcept>
#include <system_error>

/**
 * @brief Exceptions 异常
 * 异常是传递错误的类型, 当错误发生时, 抛出一个异常;
 * 抛出异常之后, 就进入了"飞行状态", 程序停止正常的执行过程, 并寻找能够管理该异常的异常处理程序.
 * !在这个过程中，离开作用域的对象会被销毁.
 * 在没有很好的方法处理本地错误的情况下, 比如在构造函数中, 一般会使用异常机制,
 * 在这种情况下, 异常在管理对象生命周期中起着至关重要的作用.
 * 传递错误的另一种方法是将错误码作为函数原型的一部分返回,
 * 在发生可以在本地处理的错误或预计在程序正常执行过程中会发生的错误的情况下, 一般会返回错误码.
 * 
 * ----- throw关键字
 * 要抛出异常, 需使用 throw 关键字, 后面跟着一个可抛出对象, 大多数对象都是可抛出的,
 * 但最好使用 stdlib 中的异常之一, 比如 <stdexcept>头文件中的 std:.runtime_error,
 * runtime_error 构造函数接受一个结尾为null的 const char* 型字符来描述错误的性质,
 * 通过 what 方法可以获取这个信息, 该方法不需要参数.
 * 
 * ------ 使用try-catch代码块
 * 使用 try-catch 块为代码块建立异常处理程序, 在 try 块中放置可能抛出异常的代码,
 * 在catch 块中为每一种可以处理的异常类型指定处理程序.
 * 
 * ------ 异常机制使用继承关系来决定处理程序是否捕获异常,
 * 处理程序将捕获给定的类型和它的任何父类型.
 * Exceptions use these inheritance relationships to determine whether
 * a handler catches an exception. Handlers will catch a given type and
 * any of its parents' types.
 * 
 * ------ stdlib Exception Classes
 * stdlib异常类, 简单的现有异常类型的层次结构可供使用
 * stdlib在<stdexcept>头文件中提供了标准异常类, 在实现异常时,这些类应该是你的首选调用对象;
 * 所有标准异常类的父类都是 std::exception 类, 所有子类都可以划分为:
 * 1. 逻辑错误 logic_error: domain_error, invalid_argument, length_error, out_of_range
 * 2. 运行时错误 runtime_error: system_error, underflow_error, overflow_error
 * 3. 语言支持错误:
 * 4. 其他继承的异常类: bad_cast, bad_alloc
 * 
 * ---- 逻辑错误, 逻辑错误来源于 logic_error 类,
 * 一般来说可以通过仔细编码来避免这些异常, 一个主要例子是类的逻辑前置条件不满足, 如当类不变量不能满足时.
 * 由于类不变量是由程序员定义的, 所以无论是编译器还是运行时环境都不能独立保证这一点,
 * 可以使用类的构造函数来检查各种条件, 如果不能构建类不变量, 则可以抛出一个异常,
 * 如果失败的原因是向构造函数传递了不正确的参数, 那么 logic_error 是一个适合抛出的异常.
 * logic_error 有几个子类应该多加注意:
 * 1. domain_error 报告与有效输入范围有关的错误, 特别是对于数学函数;
 * 2. invalid_argument 报告意外的参数错误;
 * 3. length_error 报告某些操作违反了最大尺寸约束;
 * 4. out_of_range 报告某些值不在预期范围内, 典型的例子是在数据结构中进行边界检查.
 * 
 * ---- 运行时错误, 运行时错误来源于 runtime_error 类, 这些异常可以报告程序作用域之外的错误.
 * runtime_error 也有一些很有用的子类:
 * 1. system_error 报告操作系统遇到的一些错误, 可以从这种异常中得到很多信息,
 *    <system_error>头文件中有大量的错误码和错误条件, 
 *    当 system_error被构造出来时, 错误信息会被打包进去, 这样就可以确定错误的根源,
 *    code()方法返回类型为std::errc的枚举类, 这个类包含大量的错误码, 
 *    如bad_file_descriptor, timed_out和permission_denied
 * 2. overflow_error 和 underflow_error 分别报告算术上溢出和下溢出;
 * 3. 其他错误直接继承自 exception 类, 一个常见的异常是 bad_alloc, 它报告 new 未能成功分配所需动态内存.
 * 4. 语言支持错误, 之所以存在是为了表明某些核心语言功能在运行时失效.
 * 
 * =======异常处理Handling Exceptions 
 * 异常处理的规则是基于类继承机制的, 当一个异常被抛出时, 
 * 如果抛出的异常的类型与 catch 处理程序的异常类型匹配, 
 * 或者抛出的异常的类型继承自 catch 处理程序的异常类型,那么 catch 代码块就会处理该异常.
 * 
 */

struct Groucho
{
    static void forget(int x)
    {
        if (x == 0xFACE)
        {
            throw std::runtime_error{"I'd be glad to make an exception."};
        }
        printf("Forgot 0x%x\n", x);
    }
};

class DemoClass
{
public:
    static int getValue() noexcept
    {
        return m_val;
    }

private:
    static int m_val;
};

int DemoClass::m_val = 42;

// ----------------------------------
int main(int argc, const char **argv)
{
    Groucho::forget(42);
    // Groucho::forget(64206);

    try
    {
        Groucho::forget(0xC0DE);
        Groucho::forget(0xFACE);
        Groucho::forget(0xC0FFEE);
    }
    catch (const std::runtime_error &except)
    {
        printf("exception caught with message: %s\n", except.what());
    }

    // 下面的处理程序可以捕获任何继承自 std::exception 的异常, 包括 std::logic_error
    try
    {
        throw std::logic_error{
            "It's not about who wrong "
            "it's not about who right"};
    }
    catch (std::exception &ex)
    {
        // Handles std::logic_error as it inherits from std::exception
    }

    // 下面的特殊处理程序可以捕获任何异常, 无论其类型如何
    try
    {
        throw 'z'; // Don't do this.
    }
    catch (...)
    {
        // Handles any exception, even a 'z'
    }
    // !特殊处理程序通常被用作安全机制,用于记录程序的灾难性错误,这种错误往往发生在处理某种特定类型的异常时.

    // 链接多个 catch 语句可以处理来自同 try 代码块的不同类型的异常
    try
    {
        // Code that might throw an exception
    }
    catch (const std::logic_error &ex)
    {
        // Log exception and terminate the program; there is a programming error!
    }
    catch (const std::runtime_error &ex)
    {
        // Do our best to recover gracefully
    }
    catch (const std::exception &ex)
    {
        // This will handle any exception that derives from std:exception
        // that is not a logic_error or a runtime_error.
    }
    catch (...)
    {
        // Panic; an unforeseen exception type was thrown
    }

    /**
     * @brief 重新抛出异常
     * 在 catch 代码块中, 可以使用 throw 关键字来继续寻找合适的异常处理程序,
     * 这叫作重新抛出异常, 在某些不寻常但又很重要的情况下, 在处理异常之前, 可能需要进一步检查它.
     * 
     * 与其重新抛出,不如定义新的异常类型, 并为 EACCES 错误创建一个单独的 catch 处理程序.
     */
    try
    {
        // Some code that might throw a system_error
    }
    catch (const std::system_error &ex)
    {
        if (ex.code() != std::errc::permission_denied)
        {
            // Not a permission denied error
            throw;
        }
        // Recover from a permission denied
    }

    // ==================================================
    /**
     * @brief 用户定义的异常
     * 可以随时定义自己的异常, 通常用户定义的异常继承自 std::exception,
     * 所有来自stdlib的类都使用派生自 std::exception 的异常,
     * 这使得可以很容易地用一个 catch 代码块捕获所有的异常, 无论是来自自己代码的异常还是来自stdlib的异常.
     * 
     * ----- noexcept关键字
     * 关键字 noexcept 是与异常相关的术语, 可以而且应该将不可能抛出异常的函数标记为 noexcept,
     * 标记为 noexcept 的函数创造了一个严格的契约, 明确知道这个函数绝不会抛出异常,
     * 因为编译器不会检查任何错误, 因为C++运行时将直接调用函数 std::terminate, 默认会通过 abort 退出程序;
     * 函数标记为 noexcept 可以帮助优化代码, 这些优化要求函数不抛出异常, 编译器可以自由地使用移动语义
     * 
     */
    printf("value: %d", DemoClass::getValue());

    /**
     * @brief 调用栈和异常
     * 调用栈是一个运行时结构, 它存储了当前正运行的函数信息, 当一段代码(调用者)调用一个函数(被调用者)时,
     * 机器将记录谁调用了谁并将信息放到调用栈上, 这使得程序可以有很多函数调用嵌套在一起,
     * 被调用者可以作为调用者反过来调用另一个函数.
     *
     * 栈(stack)是一种灵活的容器, 可以容纳的元素数量动态变化,
     * 所有的栈都支持两种基本操作: 将元素压到栈的顶部和将这些元素弹出, 是一种后进先出的数据结构.
     * 
     * 顾名思义, 调用栈在功能上与它的名字类似, 每当函数被调用时, 
     * 关于函数调用的信息都会被存放到栈帧中, 并压到调用栈顶, 
     * 由于每调用一次函数, 都会有一个新的栈帧(stack frame)压入栈顶, 
     * 所以被调用者可以自由地调用其他函数，形成任意深度的调用链;
     * 每当函数返回时, 它的栈帧就会从调用栈的顶部弹出, 程序就会按照之前的栈帧信息继续执行.
     * 
     * ---- 调用栈和异常处理
     * 运行时会寻找最接近抛出的异常的异常处理程序,如果当前的栈帧中有匹配的异常处理程序,它将处理该异常;
     * 如果没有找到匹配的异常处理程序, 运行时将展开调用栈, 直到找到合适的处理程序,
     * 这个过程中, 任何生命周期要结束的对象都会正常地被销毁.
     *
     * ----- 在析构函数中抛出异常
     * 如果在析构函数中抛出异常, 那么就是在玩火, 这样的异常必须在析构函数内部被捕获,
     * 假设析构函数中抛出了异常, 在栈展开过程中, 另一个异常被执行正常清理过程的另一个析构函数抛出,
     * 现在就有两个异常了, C++运行时应该如何处理这种情况呢?
     * 对于这种情况, C++运行时(runtime-library)会直接调用 terminate
     * 所以, 要把析构函数当作 noexcept 函数.
     *
     * https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
     */
    struct CyberdyneSeries
    {
        CyberdyneSeries()
        {
            printf("I'm a friend of Sarah Connor.\n");
        }

        ~CyberdyneSeries()
        {
            // throw std::runtime_error{"I'll be back.\n"};
            printf("I'll be back.\n");
        }
    };

    try
    {
        CyberdyneSeries t800;
        throw std::runtime_error{"Come with me if you want to live.\n"};
    }
    catch (const std::exception &e)
    {
        printf("Caught exception: %s\n", e.what());
    }

    return 0;
}
