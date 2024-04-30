/**
 * @file 09_fullClass.cpp
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
#include <utility>

/**
 * @brief 编译器生成的方法
 * 有五种方法可以控制移动行为和复制行为:
 * 1. 析构函数 The destructor;
 * 2. 复制构造函数 The copy constructor;
 * 3. 移动构造函数 The move constructor;
 * 4. 复制赋值运算符 The copy assignment operator;
 * 5. 移动赋值运算符 The move assignment operator;
 * 在某些情况下, 编译器可以为每个方法生成默认实现, 
 * !不幸的是, 生成方法的规则很复杂, 而且在不同的编译器中可能是不一致的.
 * 可以通过将这些方法设置为 default/delete 或通过恰当的方式实现它们来消除这种复杂性,
 * 这个一般规则叫 "五法则"(rule of five), 因为有五个方法需要指定,
 * 显式处理它们会花费点时间, 但可以在未来减少很多麻烦.
 * 
 * *总结自己实现的五个函数和编译器生成的每个函数之间的相互作用:
 * 1. 如果什么都不提供, 编译器将生成五个函数
 * (析构函数,复制构造函数,复制赋值运算符,移动构造函数,移动赋值运算符), 这就是"零法则"(rule of zero).
 * 2. 如果明确定义了任何一个函数(析构函数,复制构造函数,复制赋值运算符),就会得到所有这三个函数,
 *  这是很危险的, SimpleString 所演示的那样: 很容易陷入一种意外的情况, 即编译器基本上把所有的移动操作都转换为复制操作.
 * 3. 如果只为类提供移动语义, 编译器不会自动生成任何东西, 除了析构函数.
 * 
 */

class SimpleString
{
public:
    // *constructor
    SimpleString(size_t max_size)
        : max_size{max_size}
        , length{}
    {
        if (max_size == 0)
        {
            throw std::runtime_error{"Max size must be at least 1."};
        }
        buffer    = new char[max_size];
        buffer[0] = 0;
    }

    // *deconstructor
    ~SimpleString()
    {
        delete[] buffer;
    }

    // *copy constructor
    SimpleString(const SimpleString &other)
        : max_size{other.max_size}
        , buffer{new char[other.max_size]}
        , length{other.length}
    {
        std::strncpy(buffer, other.buffer, max_size);
    }

    // *move constructor
    SimpleString(SimpleString &&other) noexcept
        : max_size(other.max_size)
        , buffer(other.buffer)
        , length(other.length)
    {
        other.length   = 0;
        other.buffer   = nullptr;
        other.max_size = 0;
    }

    // *copy assignment operator
    SimpleString &operator=(const SimpleString &other)
    {
        if (this == &other)
            return *this;
        const auto new_buffer = new char[other.max_size];
        delete[] buffer;
        buffer   = new_buffer;
        length   = other.length;
        max_size = other.max_size;
        std::strncpy(buffer, other.buffer, max_size);
        return *this;
    }

    // *move assignment operator
    SimpleString &operator=(SimpleString &&other) noexcept
    {
        if (this == &other)
            return *this;
        delete[] buffer;
        buffer         = other.buffer;
        length         = other.length;
        max_size       = other.max_size;
        other.buffer   = nullptr;
        other.length   = 0;
        other.max_size = 0;
        return *this;
    }

    void print(const char *tag) const
    {
        printf("%s: %s", tag, buffer);
    }

    bool append_line(const char *x)
    {
        const auto x_len = strlen(x);
        if (x_len + length + 2 > max_size)
            return false;
        std::strncpy(buffer + length, x, max_size - length);
        length += x_len;
        buffer[length++] = '\n';
        buffer[length]   = 0;
        return true;
    }

private:
    size_t max_size;
    char  *buffer;
    size_t length;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    auto p_str = new SimpleString{42};

    p_str->append_line("the full standard class");
    p_str->print("Ithaca");

    if (p_str)
    {
        delete p_str;
        p_str = nullptr;
    }

    SimpleString a{50};
    a.append_line("We apologize for the");
    SimpleString b{50};
    b.append_line("Last message");
    a.print("a");
    b.print("b");
    b = std::move(a);
    // a is "moved-from"
    b.print("b");

    return 0;
}
