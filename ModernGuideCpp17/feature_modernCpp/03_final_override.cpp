/**
 * @file 03_final_override.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

class Base
{
public:
    virtual void test() const
    {
        std::cout << "Hell [Base] test...\n";
    }
};

class Child final : public Base
{
public:
    void test() const override final
    {
        std::cout << "Hell [Child] test...\n";
    }
};

// class GrandChild : public Child /* error,  语法错误,不允许重写 */
class GrandChild : public Base
{
public:
    // void test() /* error, 语法错误,不允许重写 */
    void test() const override
    {
        std::cout << "Hell [GrandChild] test...\n";
    }
};

// C++11 add 'final' keyword 来限制某个类不能被继承，或者某个虚函数不能被重写.
// C++11 add 'override' keyword 确保在派生类中声明的重写函数与基类的虚函数有相同的签名,
// 同时也明确表明将会重写基类的虚函数,保证重写的虚函数的正确性, 也提高了代码的可读性.
// -------------------------------
int main(int argc, char **argv)
{
    Base       obj_base;
    obj_base.test();

    Child      obj_child;
    obj_child.test();

    GrandChild obj_grandchild;
    obj_grandchild.test();

    return 0;
}