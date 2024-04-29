/**
 * @file 04_simpleString.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>

/**
 * @brief 扩展的例子来探讨一下构造函数, 析构函数, 成员和异常是如何融合在一起的.
 * 
 */
class SimpleString
{
public:
    SimpleString(size_t max_size)
        : m_max_size{max_size}
        , m_length{}
    {
        /**
         * @brief Because max_size is a size_t, 
         * it is unsigned and cannot be negative, 
         * so you don’t need to check for this bogus condition.
         */
        if (max_size == 0)
        {
            throw std::runtime_error{"Max size must be at least 1.\n"};
        }

        m_buffer    = new char[m_max_size];
        m_buffer[0] = 0;
    }

    ~SimpleString()
    {
        /**
         * @brief 这种模式称为资源获取即初始化(RAII)或构造函数获取,
         * 析构函数释放(ConstructorAcquires, Destructor Releases, CADRe).
         * NOTE: SimpleString 类仍然有一个隐式定义的复制构造函数,
         * 虽然它可能永远不会泄漏资源, 但如果这个类的对象被复制了, 它将有可能重复释放资源.
         * 
         */
        if (m_buffer) // *double free error
        {
            delete[] m_buffer;
            m_buffer = nullptr;
        }
    }

    void print(const char *tag) const noexcept
    {
        printf("%s: %s", tag, m_buffer);
    }

    bool append_line(const char *str)
    {
        // signature: size_t strlen(const char* str);
        const auto str_len = strlen(str);
        if (str_len + m_length + 2 > m_max_size)
            return false;

        // signature: char* std::strncpy(char* destination, const char* source, std::size_t num);
        std::strncpy(m_buffer + m_length, str, m_max_size - m_length);
        m_length += str_len;

        // 在 buffer 的最后添加一个换行符 \n 和一个null字符
        m_buffer[m_length++] = '\n';
        m_buffer[m_length]   = 0;
        return true;
    }

private:
    size_t m_max_size;
    char  *m_buffer;
    size_t m_length;
};

// Composing a SimpleString
/**
 * @brief 表明了对象的成员在构造过程中的顺序:
 * 成员在类的构造函数之前被构造, 如果不知道类的成员的不变量, 又怎么能建立类的不变量呢?
 * 正如预期的那样x的成员 m_str 被正确地创建了,
 * 因为对象成员的构造函数是在对象自己的构造函数之前被调用的,
 * 结果就得到了 Constructed:x 的消息。
 * 此时成员 m_str 仍然有效, 因为成员析构函数是在类对象的析构函数之后调用的.
 * 
 */
class SimpleStringOwner
{
public:
    SimpleStringOwner(const char *x)
        : m_str{10}
    {
        if (!m_str.append_line(x))
            throw std::runtime_error{"Not enough memory!\n"};

        m_str.print("Constructed: ");
    }

    ~SimpleStringOwner()
    {
        m_str.print("About to destroy: ");
    }

private:
    SimpleString m_str;
};

// function call stack unwinding
void fn_c()
{
    SimpleStringOwner c{"=cccccccccc"}; // !exception
}

void fn_b()
{
    SimpleStringOwner b{"=b"};
    fn_c();
}

/**
 * @brief 声明一个POD对象, 采用设计模式为工厂方法, 因为它们的目的是初始化对象.
 * 
 */

struct HumptyDumpty
{
    HumptyDumpty() = default;
    bool is_together_again();
};

struct Result
{
    HumptyDumpty hd;
    bool         success;
};

Result make_humpty()
{
    HumptyDumpty hd{};
    bool         is_valid{};
    // Check that hd is valid and set is_valid appropriately
    return {hd, is_valid};
}

bool send_kings_horses_and_men()
{
    auto [hd, success] = make_humpty();
    if (!success)
        return false;
    // Class invariants established
    return true;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    SimpleString string{115};

    string.append_line("Starbuck, whaddya hear?");
    string.append_line("Nothin' but the rain.");
    string.print("A: ");
    string.append_line("Grab your gun and bring the cat in.");
    string.append_line("Aye-aye sir, coming home.");
    string.print("B: ");

    // *作为使用者, 有责任检查这个条件
    if (!string.append_line("Galactica!"))
    {
        printf("String was not big enough to append another message.");
    }

    SimpleStringOwner x{"x"};
    printf("x is alive\n");

    /**
     * @brief Call Stack Unwinding 调用栈展开, 异常处理和调用栈展开的工作方式
     * *看看调用栈是什么样的, 哪些对象是有效的, 异常处理情况是什么样的.
     * 1. 现在有一个异常需要处理, 调用栈将展开, 直到找到合适的处理程序才停下来,
     *   !所有因调用栈展开而离开作用域的对象都将被销毁.
     * 2. 一旦 try{} 块中发生异常, 就不会再有其他语句执行,
     *    因此 d永远不会初始化, 也永远看不到 d的构造函数打印到控制台.
     *   调用栈展开后, 程序将立即执行到 catch 块.
     * 
     */
    try
    {
        SimpleStringOwner a{"=a"};
        fn_b();
        SimpleStringOwner d{"=d"};
    }
    catch (const std::exception &e)
    {
        printf("Exception: %s\n", e.what());
    }

    /**
     * @brief Exceptions and Performance 异常和性能
     * 在程序中必须处理错误, 错误是不可避免的, 当正确地使用异常机制, 并且没有错误发生时,
     * 代码会比采用手工错误检查的代码更快; 如果确实发生了错误, 异常处理有时可能会慢一些,
     *  但在代码健壮性和可维护性方面, 会获得巨大的收益.
     * *异常处理可以使程序在正常执行时速度更快, 而在执行失败时行为更好.
     * 当C++程序正常执行时(没有抛出异常), 没有与检查异常相关的运行时开销, 只有当异常被抛出时才会有开销.
     * 
     * 异常在通常的C++程序中的核心作用, 有时无法使用异常, 例如嵌入式开发需要实时性保障;
     * 在这种情况下, 根本就没有工具(暂时还没有).
     * 另一个例子是遗留代码, 异常之所以优雅, 是因为它们与RAII对象配合得很好,
     * 当析构函数负责清理资源时, 栈展开是防止资源泄漏的一种直接而有效的方法.
     * 在遗留代码中, 可能会发现手动资源管理和错误处理代替了RAI对象, 这使得使用异常变得非常危险,
     * !因为只有在RAII对象的配合下, 栈展开才是安全的, 如果没有它们, 很容易泄漏资源.
     * 
     */

    /**
      * @brief Alternatives to Exceptions 异常的替代方法
      * 在异常不可用的情况下, 虽然需要手动检查错误, 但仍可以利用一些有用的C++特性来减少错误.
      * 首先可以通过暴露一些方法来手动约束类不变量, 让这些方法判断类不变量是否可以建立.
      * 在典型C++中, 只会在构造函数中抛出异常, 
      * 但在们必须记住要在调用代码中检查这种情况并将其作为一个错误条件.
      * 
      * 第二种应对策略是使用结构化绑定声明返回多个值, 这是一种语言特性,
      * 它允许从一个函数调用中返回多个值, 
      * 可以使用这个特性在返回正常的返回值的同时返回 success 标志.
      * 
      */
    if (send_kings_horses_and_men())
        printf("return is true\n");
    else
        printf("return is false\n");

    return 0;
}
