/**
 * @file 03_final_override.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. C++中增加了final关键字来限制某个类不能被继承, 或者某个虚函数不能被重写,
  * *如果使用final修饰函数, 只能修饰虚函数, 并且要把final关键字放到类或者函数的后面.
  * 
  * 2. override关键字确保在派生类中声明的重写函数与基类的虚函数有相同的签名,
  * *同时也明确表明将会重写基类的虚函数, 这样就可以保证重写的虚函数的正确性, 提高代码的可读性
  * 和final一样这个关键字要写到方法的后面.
  * 
  * ?使用了override关键字之后,假设在重写过程中因为误操作,
  * ?写错了函数名或者函数参数或者返回值编译器都会提示语法错误,提高了程序的正确性,降低了出错的概率.
  * 
  */

#include <iostream>

// 如果使用final修饰函数，只能修饰虚函数，这样就能阻止子类重写父类的这个函数了
class BaseClass
{
public:
    virtual void test()
    {
        std::cout << "Base class...\n";
    }
};

class ChildClass : public BaseClass
{
public:
    void test() final
    {
        std::cout << "Child class...\n";
    }
};

class GrandChild : public ChildClass
{
public:
    // !语法错误, 不允许重写, 该虚函数已经被父类 final 了
    // void test()
    // {
    //     std::cout << "GrandChild class...";
    // }
};

// 使用final关键字修饰过的类是不允许被继承的，也就是说这个类不能有派生类
class Base
{
public:
    virtual void test()
    {
        std::cout << "Base class...\n";
    }
};

class Child final : public Base
{
public:
    void test()
    {
        std::cout << "Child class...\n";
    }
};

// !error, 语法错误, 该父类已经被 final 了
// class GrandChild : public Child
// {
// public:
// };

class Interface
{
public:
    Interface()
    {
        std::cout << "Interface() called...\n";
    }

    virtual ~Interface()
    {
        std::cout << "~Interface() called...\n";
    }

    virtual void print()
    {
        std::cout << "Base interface class...\n";
    }
};

class Instance : public Interface
{
public:
    Instance()
    {
        std::cout << "Instance() called...\n";
    }

    virtual ~Instance()
    {
        std::cout << "~Instance() called...\n";
    }

    void print() override
    {
        std::cout << "Child Instance class...\n";
    }
};

class GrandInstance : public Instance
{
public:
    GrandInstance()
    {
        std::cout << "GrandInstance() called...\n";
    }

    virtual ~GrandInstance()
    {
        std::cout << "~GrandInstance() called...\n";
    }

    void print() override
    {
        std::cout << "Grand Instance Child class...\n";
    }
};

// -----------------------------------
int main(int argc, const char **argv)
{
    Interface *obj1 = new Interface{};
    Interface *obj2 = new Instance{};
    Interface *obj3 = new GrandInstance{};

    obj1->print();
    obj2->print();
    obj3->print();

    if (nullptr == obj1)
    {
        delete obj1;
        obj1 = nullptr;
    }
    if (nullptr == obj2)
    {
        delete obj2;
        obj2 = nullptr;
    }
    if (nullptr == obj3)
    {
        delete obj3;
        obj3 = nullptr;
    }

    return 0;
}
