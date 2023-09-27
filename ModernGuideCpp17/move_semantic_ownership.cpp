/**
 * @file move_semantic_ownership.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

struct X
{
    X()
    {
        puts("X()");
    }

    X(const X &)
    {
        puts("X(const X&)");
    }

    X(X &&) noexcept
    {
        puts("X(X&&)");
    }

    ~X()
    {
        puts("~X()");
    }
};

class Test
{
    X *m_p;

public:
    Test()
        : m_p{nullptr}
    {
    }

    Test(X *x)
        : m_p{x}
    {
    }

    Test(const Test &t)
    { //单纯的拷贝新的对象。
        m_p = new X(*t.m_p);
    }

    Test(Test &&t) noexcept
    { //转移所有权，即转移拥有的指向实际数据的指针，无拷贝
        m_p   = t.m_p;
        t.m_p = nullptr;
    }

    ~Test()
    {
        if (m_p != nullptr)
        { //为空代表无所有权，也不需要delete
            delete m_p;
        }
    }

    constexpr bool empty() const noexcept
    {
        return m_p == nullptr;
    }
};

Test func()
{
    Test t{new X};
    puts("---------");
    return t;
}

/**
 * @brief C++所有权与移动语义, 在 STL 的容器设计均有体现(seen source code)
 * 
 * Quote from https://zhuanlan.zhihu.com/p/658035687
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    {
        Test t{new X};
        std::cout << t.empty() << '\n'; //打印0
        puts("---------");
        Test t2{std::move(t)};          //移动构造 转移所有权
        std::cout << t.empty() << '\n'; //打印1，表示所有权已经被转移，即t不再拥有指向实际数据X的指针
    }
    puts("---------------------");
    {
        Test t{new X};
        std::cout << t.empty() << '\n'; //打印0
        puts("---------");
        Test t2{t};                     //拷贝构造
        std::cout << t.empty() << '\n'; //打印0
    }
    puts("----------------------");
    {
        auto result = func();
    }

    return 0;
}