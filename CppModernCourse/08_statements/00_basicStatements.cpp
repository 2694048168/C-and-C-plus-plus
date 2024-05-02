/**
 * @file 00_basicStatements.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 表达式语句是后跟分号(;)的表达式; expression statement
 * 复合语句也称为程序块(block), 是用大括号 {} 括起来的一系列语句; Compound Statements
 * 
 */

struct Tracer
{
    Tracer(const char *name)
        : m_name{name}
    {
        printf("%s constructed.\n", m_name);
    }

    ~Tracer()
    {
        printf("%s destructed.\n", m_name);
    }

private:
    const char *const m_name;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    int x{};
    ++x; // expression statement
    printf("the value of x: %d", x);

    /**
     * @brief 每个块都声明一个新的作用域, 称为块作用域;
     * 在块作用域内声明的具有自动存储期的对象的生命周期受该块的约束,
     * 在块中声明的变量将以明确定义的顺序销毁, 即按声明顺序相反的顺序销毁;
     * 
     */
    Tracer main{"main"};
    {
        printf("Block a\n");
        Tracer a1{"a1"};
        Tracer a2{"a2"};
    }
    // !请注意, 这两个Tracer以与其初始化顺序相反的顺序销毁, 即先销毁 a2 再销毁 a1
    {
        printf("Block b\n");
        Tracer b1{"b1"};
        Tracer b2{"b2"};
    }

    return 0;
}
