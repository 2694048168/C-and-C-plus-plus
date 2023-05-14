/**
 * @file derive.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Class Inheritance
 * @attention
 *
 */

#include <iostream>
#include <memory>
#include <string>

class Base
{
public:
    int a;
    int b;
    Base(int a = 0, int b = 0)
    {
        this->a = a;
        this->b = b;
        std::cout << "Constructor Base::Base(" << a << ", " << b << ")\n";
    }
    ~Base()
    {
        std::cout << "Destructor Base::~Base()\n";
    }

    int product()
    {
        return a * b;
    }

    friend std::ostream &operator<<(std::ostream &os, const Base &obj)
    {
        os << "Base: a = " << obj.a << ", b = " << obj.b;
        return os;
    }
};

class Derived : public Base
{
public:
    int c;
    Derived(int c) : Base(c - 2, c - 1), c(c)
    {
        this->a += 3; // it can be changed after initialization
        std::cout << "Constructor Derived::Derived(" << c << ")\n";
    }
    ~Derived()
    {
        std::cout << "Destructor Derived::~Derived()\n";
    }

    int product()
    {
        return Base::product() * c;
    }

    friend std::ostream &operator<<(std::ostream &os, const Derived &obj)
    {
        // call the friend function in Base class
        os << static_cast<const Base &>(obj) << std::endl;

        os << "Derived: c = " << obj.c;
        return os;
    }
};

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    {
        Base base(1, 2);
        std::cout << "Product = " << base.product() << std::endl;
        std::cout << base << std::endl;
    }

    std::cout << "----------------------\n";
    
    {
        Derived derived(5);
        std::cout << derived << std::endl;
        std::cout << "Product = " << derived.product() << std::endl;
    }

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ derive.cpp
 * $ clang++ derive.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */