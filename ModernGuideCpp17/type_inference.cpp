/**
 * @file type_inference.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <typeinfo>

int func1()
{
    return 42;
}

char func2()
{
    return 'G';
}

const char *func3()
{
    return "hello decltype";
}

/**
 * @brief auto and decltype for type inference in C++.
 * @return decltype(a < b ? a : b) 
 */
template<typename T1, typename T2>
auto find_minimum_value(T1 a, T2 b) -> decltype(a < b ? a : b)
{
    // TODO Reference to stack memory returned?
    return (a < b) ? a : b;
}

/**
 * @brief the type inference in modern C++.
 * 'auto' | 'decltype' | 'typeid'
 * Decltype gives the type information at compile time
 * while typeid gives at runtime.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    /* 'auto' for type inference in C++
    ------------------------------------- */
    auto x    = 4;
    auto y    = 3.37;
    auto z    = 3.37f;
    auto c    = 'a';
    auto ptr  = &x;
    auto pptr = &ptr; //pointer to a pointer

    std::cout << typeid(x).name() << "\n"
              << typeid(y).name() << "\n"
              << typeid(z).name() << "\n"
              << typeid(c).name() << "\n"
              << typeid(ptr).name() << "\n"
              << typeid(pptr).name() << std::endl;

    /* 'decltype' for type inference in C++
    ----------------------------------------- */
    std::cout << "--------------------------" << std::endl;
    decltype(func1()) seed{42};
    decltype(func2()) ch{'L'};
    decltype(func3()) str{"weili_yzzcq@163.com"};

    std::cout << typeid(seed).name() << '\n' << typeid(ch).name() << '\n' << typeid(str).name() << std::endl;

    /* example for type inference in C++
    ----------------------------------------- */
    std::cout << "--------------------------" << std::endl;
    std::cout << find_minimum_value(4, 3.14) << std::endl;
    std::cout << find_minimum_value(5.4f, 4.2f) << std::endl;
    std::cout << find_minimum_value(42, 24) << std::endl;

    return 0;
}