/**
 * @file 13_delegate_inheritance_constructor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 委托构造和继承构造函数
 * @version 0.1
 * @date 2024-09-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. 委托构造函数
 * 委托构造函数允许使用同一个类中的一个构造函数调用其它的构造函数, 从而简化相关变量的初始化.
 * 
 * 2. 继承构造函数
 * C++11中提供的继承构造函数可以让派生类直接使用基类的构造函数,而无需自己再写构造函数,
 * 尤其是在基类有很多构造函数的情况下,可以极大地简化派生类构造函数的编写.
 * 
 */

#include <iostream>
#include <string>

class Test
{
public:
    Test() {};

    Test(int max)
    {
        this->m_max = max > 0 ? max : 100;
    }

    // Test(int max, int min)
    // {
    //     this->m_max = max > 0 ? max : 100; // 冗余代码
    //     this->m_min = min > 0 && min < max ? min : 1;
    // }

    // Test(int max, int min, int mid)
    // {
    //     this->m_max    = max > 0 ? max : 100;            // 冗余代码
    //     this->m_min    = min > 0 && min < max ? min : 1; // 冗余代码
    //     this->m_middle = mid < max && mid > min ? mid : 50;
    // }

    // 在C++11之前构造函数是不能调用构造函数的，加入了委托构造之后
    // 并且在一个构造函数中调用了其他的构造函数用于相关数据的初始化,相当于是一个链式调用.
    // 在使用委托构造函数的时候还需要注意一些几个问题:
    // ?1. 这种链式的构造函数调用不能形成一个闭环(死循环),否则会在运行期抛异常;
    // ?2. 如果要进行多层构造函数的链式调用,建议将构造函数的调用的写在初始列表中而不是函数体内部,否则编译器会提示形参的重复定义;
    // ?3. 在初始化列表中调用了代理构造函数初始化某个类成员变量之后,就不能在初始化列表中再次初始化这个变量了;
    Test(int max, int min)
        : Test(max)
    {
        this->m_min = min > 0 && min < max ? min : 1;
    }

    Test(int max, int min, int mid)
        : Test(max, min)
    {
        this->m_middle = mid < max && mid > min ? mid : 50;
    }

    int m_min;
    int m_max;
    int m_middle;
};

class Base
{
public:
    Base(int i)
        : m_i(i)
    {
    }

    Base(int i, double j)
        : m_i(i)
        , m_j(j)
    {
    }

    Base(int i, double j, std::string k)
        : m_i(i)
        , m_j(j)
        , m_k(k)
    {
    }

    int         m_i;
    double      m_j;
    std::string m_k;
};

// ====== 不使用继承构造函数, 则子类需要重新写每一个构造函数
class Child : public Base
{
public:
    Child(int i)
        : Base(i)
    {
    }

    Child(int i, double j)
        : Base(i, j)
    {
    }

    Child(int i, double j, std::string k)
        : Base(i, j, k)
    {
    }
};

// ====== 不使用继承构造函数, 则子类需要重新写每一个构造函数
class Child_ : public Base
{
public:
    // 继承构造函数的使用方法是这样的: 通过使用using 类名::构造函数名
    // (其实类名和构造函数名是一样的)来声明使用基类的构造函数,
    // 这样子类中就可以不定义相同的构造函数了,直接使用基类的构造函数来构造派生类对象.
    using Base::Base;
};

// 另外如果在子类中隐藏了父类中的同名函数,也可以通过using的方式在子类中使用基类中的这些父类函数
class Base__
{
public:
    Base__(int i)
        : m_i(i)
    {
    }

    Base__(int i, double j)
        : m_i(i)
        , m_j(j)
    {
    }

    Base__(int i, double j, std::string k)
        : m_i(i)
        , m_j(j)
        , m_k(k)
    {
    }

    void func(int i)
    {
        std::cout << "base class: i = " << i << std::endl;
    }

    void func(int i, std::string str)
    {
        std::cout << "base class: i = " << i << ", str = " << str << std::endl;
    }

    int         m_i;
    double      m_j;
    std::string m_k;
};

class Child__ : public Base__
{
public:
    using Base__::Base__;
    using Base__::func;

    // 子类中的func()函数隐藏了基类中的两个func(),
    // 因此默认情况下通过子类对象只能调用无参的func(),
    // 子类代码中添加了using Base::func;之后,
    // 就可以通过子类对象直接调用父类中被隐藏的带参func()函数了
    void func()
    {
        std::cout << "child class: i'am Ithaca!\n";
    }
};

// -------------------------------------
int main(int argc, const char **argv)
{
    Test t(90, 30, 60);
    std::cout << "min: " << t.m_min << ", middle: " << t.m_middle << ", max: " << t.m_max << std::endl;

    Child c(520, 13.14, "i love you");
    std::cout << "int: " << c.m_i << ", double: " << c.m_j << ", string: " << c.m_k << std::endl;

    Child_ c1(520, 13.14);
    std::cout << "int: " << c1.m_i << ", double: " << c1.m_j << std::endl;
    Child_ c2(520, 13.14, "i love you");
    std::cout << "int: " << c2.m_i << ", double: " << c2.m_j << ", string: " << c2.m_k << std::endl;

    Child__ c__(250);
    c__.func();
    c__.func(19);
    c__.func(19, "Ithaca");

    return 0;
}
