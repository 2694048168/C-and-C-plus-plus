/**
 * @file 06_copyAssignment.cpp
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
 * @brief 复制赋值 Copy Assignment
 * 在C++中生成副本的一种方法是使用复制赋值运算符,
 * 可以创建对象的副本, 并将其赋给另一个现有对象,
 * NOTE: 注意 导致未定义行为, 因为没有自定义的复制赋值运算符.
 * !警告 简单类型的默认复制赋值运算符只是将源对象中的成员复制到目标对象中.
 * 两个 SimpleString 类拥有同一个缓冲区, 这可能会导致指针悬空和重复释放.
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

// -----------------------------------
int main(int argc, const char **argv)
{
    SimpleString a{50};
    a.append_line("We apologize for the");
    SimpleString b{50};
    b.append_line("Last message");
    a.print("a");
    b.print("b");
    b = a;
    a.print("a");
    b.print("b");

    return 0;
}
