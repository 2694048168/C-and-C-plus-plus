/**
 * @file object_constructor.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief C++ 构造函数 constructor; 委托构造函数 (delegate constructor) 和继承构造函数 (Inheritance constructor)
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>

class Base
{
public:
    int value1;
    int value2;

    Base()
    {
        value1 = 1;
    }

    Base(int value) : Base() /* delegate Base() constructor */
    {
        value2 = value;
    }
};

class Subclass : public Base
{
public:
    using Base::Base; /* inheritance constructor */
};

int main(int argc, char **argv)
{
    // delegate constructor
    Base b(2);
    std::cout << b.value1 << std::endl;
    std::cout << b.value2 << std::endl;

    // inheritance constructor
    Subclass s(3);
    std::cout << s.value1 << std::endl;
    std::cout << s.value2 << std::endl;

    return 0;
}
