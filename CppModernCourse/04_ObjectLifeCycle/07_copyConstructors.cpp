/**
 * @file 07_copyConstructors.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <cstring>
#include <stdexcept>

/**
 * @brief 复制构造函数 Copy Constructors
 * 复制一个对象一种是使用复制构造函数,
 * 它将创建一个副本并将其分配给一个全新的对象, 复制构造函数看起来和其他构造函数一样.
 * NOTE: other 被指定为 const, 复制某个class时, 没有理由去修改它,
 *  可以像使用其他构造函数一样使用复制构造函数, 如使用带括号初始化列表的统一初始化语法.
 * 实现的复制构造函数, 想要的是所谓的"深复制"(deep copy),
 * 即把原始缓冲区(buffer)所指向的数据复制到新的缓冲区中, 而非 shallow copy.
 * 
 * 当按值将 SimpleString 传递到某函数中时, 便会调用复制构造函数
 * 当按值传递一个对象时, 复制构造函数会被调用,
 * NOTE: 注意不应该通过按值传递来避免修改, 而应该使用 const 引用.
 * 复制操作对性能的影响可能很大, 特别是在涉及自由存储分配和缓冲区复制的情况下,
 * 例如假设有一个管理千兆字节数据的类, 每次复制对象时, 都需要分配和复制1GB的数据,
 * 这可能会花费很长时间, 所以要确定自己的确需要复制数据,
 * 如果可以通过传递 const 引用来实现, 则强烈推荐使用这种方式.
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

    /**
     * @brief 与其复制指针 buffer, 不如在自由存储区上重新分配一个区域,
     * 然后复制原始 buffer 指向的所有数据, 这样就有了两个独立的 SimpleString 类.
     * other.buffer 指向的内容复制到 buffer 指向的数组中.
     * 
     */
    SimpleString(const SimpleString &other)
        : m_max_size{other.m_max_size}
        , m_buffer{new char[other.m_max_size]}
        , m_length{other.m_length}
    {
        // std::strncpy(m_buffer, other.m_buffer, m_max_size);
        strcpy_s(m_buffer, m_max_size, other.m_buffer);
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

    /**
     * @brief 复制赋值运算符使用 operator= 语法
     * 自定义的SimpleString复制赋值运算符,
     * 复制赋值运算符返回对结果的引用, 它的值永远都是 *this,
     * 一般来说, 检查 other 是否正好指向 this 也是很好的实践.
     *  
     * 复制赋值运算符首先分配一个大小合适的 new buffer,
     * 然后清理 buffer, 复制 buffer, length, max_size,
     * 再把 other.buffer 的内容复制到当前的 buffer
     *  
     */
    SimpleString &operator=(const SimpleString &other)
    {
        if (this == &other)
            return *this;

        const auto new_buffer = new char[other.m_max_size];
        delete[] m_buffer;
        m_buffer = new_buffer;
        strcpy_s(m_buffer, m_max_size, other.m_buffer);

        m_length   = other.m_length;
        m_max_size = other.m_max_size;

        return *this;
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

void foo(SimpleString x)
{
    x.append_line("This change is lost.");
}

void foo_ref(SimpleString &x)
{
    x.append_line("This change is lost.");
}

// -----------------------------------
int main(int argc, const char **argv)
{
    SimpleString a{50};
    a.append_line("We apologize for the");
    SimpleString a_copy{a};
    a.append_line("inconvenience.");
    a_copy.append_line("incontinence.");
    a.print("a");
    a_copy.print("a_copy");

    SimpleString str{42};
    foo(str); // Invokes copy constructor
    str.print("Still empty");

    SimpleString str_ref{42};
    foo(str_ref); // Invokes copy constructor
    str_ref.print("Ref Still empty");

    return 0;
}
