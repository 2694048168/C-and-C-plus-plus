/**
 * @file 05_memberInitializerLists.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cassert>
#include <cstdio>
#include <iostream>
#include <typeinfo>
#include <vector>

/**
 * @brief 成员初始化列表 Member Initializer Lists
 * 成员初始化列表是初始化类成员的主要机制,
 * 要声明成员初始化列表, 请在构造函数中的参数列表后放置一个冒号,
 * 然后插入一个或多个逗号分隔的成员初始化器.
 * 成员初始化器指成员名称后面跟着大括号的初始化过程,
 * 成员初始化器允许在运行时设置 const 字段的值.
 * 
 * 所有的成员初始化都在构造函数之前执行, 这有两个好处:
 * 1. 它在构造函数执行之前确保所有成员的有效性, 所以可以专注于初始化逻辑而不是成员错误检查;
 * 2. 成员只初始化一次, 如果在构造函数中重新给成员赋值, 可能会产生额外的工作量.
 * NOTE: 应该按照成员在类定义中出现的顺序来排序成员初始化列表, 因为成员的构造函数将按照这个顺序被调用.
 * 
 */

struct ClockOfTheLongNow
{
    ClockOfTheLongNow(long year)
        : m_year(year)
    {
    }

    int get_year() const
    {
        return m_year;
    }

    long m_year;
};

struct Avout
{
    Avout(const char *name, long year_of_aper)
        : m_name{name}
        , m_aper{year_of_aper}
    {
    }

    void announce() const
    {
        printf("My name is %s and my next aper is %d.\n", m_name, m_aper.get_year());
    }

    const char       *m_name;
    ClockOfTheLongNow m_aper;
};

template<typename T>
void printContainer(const T &container)
{
    printf("The element value: \n");
    for (const auto &elem : container)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    Avout obj = Avout{"Ithaca", 2035};
    obj.announce();

    /**
     * @brief auto类型推断
     * 作为一种强类型语言, C++为它的编译器提供了大量的信息.
     * 当初始化元素或返回函数返回值时, 编译器可以根据上下文推断出类型信息.
     * auto 关键字告诉编译器执行这样的推断, 这样就不用输入多余的类型信息了.
     * 
     * 当然对于基本内置类型, auto操作并不能带来多少好处,
     * 但是当类型变得更加复杂时, 例如处理来自标准库容器的迭代器时,
     * 它确实可以节省不少输入操作, 它还能让代码在重构时更灵活(替换容器或者模板类型后, 依然满足).
     * 
     */

    auto the_answer{42}; // int
    assert(typeid(int).name() == typeid(the_answer).name());

    auto foot{12L}; // long
    assert(typeid(long).name() == typeid(foot).name());

    auto rootbeer{5.0F}; // float
    assert(typeid(float).name() == typeid(rootbeer).name());

    auto cheeseburger{10.0}; // double
    assert(typeid(double).name() == typeid(cheeseburger).name());

    auto politifact_claims{false}; // bool
    assert(typeid(bool).name() == typeid(politifact_claims).name());

    auto cheese{"string"}; // char[7]
    assert(typeid(const char *).name() == typeid(cheese).name());

    /**
     * @brief auto和引用类型
     * 通常, 可以在 auto 前后添加修饰符, 如 &, *和 const,
     * 这种修饰会增加特定的含义(分别是引用、指针和 const ).
     * 在 auto 声明中添加修饰符的行为就像希望的那样: 如果添加了修饰符, 生成的类型就会保证有那个修饰符.
     * 
     * ===== auto和代码重构
     * auto 关键字有助于使代码更简单, 在重构时更灵活, 建议总是使用 auto.
     * NOTE: 使用大括号初始化时有一些极端情况要注意, 在这些情况下, 可能会得到令人惊讶的结果,
     * 但这些情况很少, 特别是在C++17修正了一些迂腐的问题之后。
     *  在C++17之前, 使用带大括号{} 的 auto 会生成一个特殊的对象, 叫作 std::initializer_list. 
     * 
     */
    std::vector<int> vec_int{32, 35, 6, 3, 9, 23, 7};
    printContainer(vec_int);

    std::vector<double> vec_double{32., 35., 6., 3., 9., 93.};
    printContainer(vec_double);

    return 0;
}
