/**
 * @file 15_move_forward.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++11添加了右值引用,并且不能使用左值初始化右值引用,
 * 如果想要使用左值初始化一个右值引用需要借助std::move()函数,
 * 使用std::move方法可以将左值转换为右值.
 * *使用这个函数并不能移动任何东西,而是和移动构造函数一样都具有移动语义,
 * ?将对象的状态或者所有权从一个对象转移到另一个对象,只是转移,没有内存拷贝.
 * 从实现上讲, std::move基本等同于一个类型转换：static_cast<T&&>(lvalue);
 * 
 * 2. forward
 * 右值引用类型是独立于值的,一个右值引用作为函数参数的形参时,
 * *在函数内部转发该参数给内部其他函数时,它就变成一个左值,并不是原来的类型了.
 * 如果需要按照参数原来的类型转发到另一个函数,可以使用C++11提供的std::forward()函数,
 * ?该函数实现的功能称之为完美转发.
 * 1. 当T为左值引用类型时，t将被转换为T类型的左值;
 * 2. 当T不是左值引用类型时，t将被转换为T类型的右值;
 * 
 */

#include <algorithm>
#include <iostream>
#include <list>
#include <string>

void print(std::string &Ele)
{
    std::cout << Ele << ",";
}

template<typename T>
void printValue(T &t)
{
    std::cout << "l-value: " << t << std::endl;
}

template<typename T>
void printValue(T &&t)
{
    std::cout << "r-value: " << t << std::endl;
}

template<typename T>
void testForward(T &&v)
{
    printValue(v);
    printValue(std::move(v));
    printValue(std::forward<T>(v));
    std::cout << std::endl;
}

// ----------------------------------
int main(int argc, const char **argv)
{
    // 假设一个临时容器很大，并且需要将这个容器赋值给另一个容器
    std::list<std::string> ls;
    ls.push_back("hello");
    ls.push_back("world");

    std::list<std::string> ls1 = ls;            // 需要拷贝, 效率低
    std::list<std::string> ls2 = std::move(ls); // 转移所有权,参考Rust中的所有权机制

    std::for_each(ls1.begin(), ls1.end(), print);
    std::for_each(ls2.begin(), ls2.end(), print);

    /*  如果不使用std::move,拷贝的代价很大,性能较低;
     使用std::move几乎没有任何代价,只是转换了资源的所有权.
     如果一个对象内部有较大的堆内存或者动态数组时,使用std::move()就可以非常方便的进行数据所有权的转移.
     另外也可以给类编写相应的移动构造函数（T::T(T&& another)）
     和具有移动语义的赋值函数（T&& T::operator=(T&& rhs)）,
     在构造对象和赋值的时候尽可能的进行资源的重复利用, 因为它们都是接收一个右值引用参数. */

    std::cout << "\n==================================\n";
    testForward(520);
    int num = 1314;
    testForward(num);
    testForward(std::forward<int>(num));
    testForward(std::forward<int &>(num));
    testForward(std::forward<int &&>(num));

    return 0;
}
