/**
 * @file 07_noexcept_throw.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 异常通常用于处理逻辑上可能发生的错误, 
 * 在C++98中提供了一套完善的异常处理机制, 可以直接在程序中将各种类型的异常抛出,从而强制终止程序的运行.
 * 1. 基本异常方式
 * 2. 异常接口声明
 * 
 * 
 */

#include <iostream>

// 异常接口声明
struct MyException
{
    MyException(std::string s)
        : msg(s)
    {
    }

    std::string msg;
};

// 显示指定可以抛出的异常类型
/* 在 divisionMethod 函数声明之后定义了一个动态异常声明 throw(MyException, int),
该声明指出了divisionMethod可能抛出的异常的类型.
事实上，该特性很少被使用，因此在C++11中被弃用了,
而表示函数不会抛出异常的动态异常声明 throw() 也被新的 noexcept 异常声明所取代.
noexcept 形如其名,表示其修饰的函数不会抛出异常.
不过与 'throw()'动态异常声明不同的是, 
在 C++11 中如果 noexcept 修饰的函数抛出了异常,
编译器可以选择直接调用 std::terminate() 函数来终止程序的运行,
这比基于异常机制的 throw() 在效率上会高一些,
这是因为异常机制会带来一些额外开销,比如函数抛出异常,会导致函数栈被依次地展开（栈解旋）
并自动调用析构函数释放栈上的所有对象.
 */
// double divisionMethod1(int a, int b) throw(MyException, int)
double divisionMethod1(int a, int b)
{
    if (b == 0)
    {
        throw MyException("division by zero!!!");
        // throw 100;
    }
    return a / b;
}

// 抛出任意异常类型
double divisionMethod2(int a, int b)
{
    if (b == 0)
    {
        throw MyException("division by zero!!!");
        // throw 100;
    }
    return a / b;
}

// 不抛出任何异常
double divisionMethod3(int a, int b) throw()
{
    if (b == 0)
    {
        std::cout << "division by zero!!!" << std::endl;
    }
    return a / b;
}

// 对于不会抛出异常的函数 since C++11
double divisionMethod(int a, int b) noexcept
{
    if (b == 0)
    {
        std::cout << "division by zero!!!" << std::endl;
        return -1;
    }
    return a / b;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    // 异常被抛出后，从进入try块起，到异常被抛掷前，这期间在栈上构造的所有对象，都会被自动析构
    // 析构的顺序与构造的顺序相反。这一过程称为栈的解旋。
    try
    {
        throw -1;
    }
    catch (int e)
    {
        std::cout << "int exception, value: " << e << std::endl;
    }
    std::cout << "That's ok!\n" << std::endl;

    // ===================================
    try
    {
        double v_0 = divisionMethod(100, 0);
        double v   = divisionMethod2(100, 0);
        std::cout << "value: " << v << std::endl;
    }
    catch (int e)
    {
        std::cout << "catch except: " << e << std::endl;
    }
    catch (MyException e)
    {
        std::cout << "catch except: " << e.msg << std::endl;
    }
    // ===================================
    // 从语法上讲，noexcept 修饰符有两种形式：
    // 简单地在函数声明后加上 noexcept 关键字, 可以接受一个常量表达式作为参数，如下所示∶
    // double divisionMethod(int a, int b) noexcept(常量表达式);
    // 常量表达式的结果会被转换成一个bool类型的值：
    // 值为 true，表示函数不会抛出异常
    // 值为 false，表示有可能抛出异常这里
    // 不带常量表达式的noexcept相当于声明了noexcept（true），即不会抛出异常。

    return 0;
}
