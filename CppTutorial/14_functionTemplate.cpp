/**
 * @file 14_functionTemplate.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习之函数模板
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

template<typename T>
void addNum(const T &number1, const T &number2, T &result)
{
    result = number1 + number2;
    std::cout << number1 << " + " << number2 << " = " << result << std::endl;
}

template<typename T>
void subNum(const T &number1, const T &number2, T &result)
{
    result = number1 - number2;
    std::cout << number1 << " - " << number2 << " = " << result << std::endl;
}

template<typename T>
void mulNum(const T &number1, const T &number2, T &result)
{
    result = number1 * number2;
    std::cout << number1 << " * " << number2 << " = " << result << std::endl;
}

// ====================================
int main(int argc, const char **argv)
{
    std::cout << "------------ template Type == int ------------\n";
    int sum_int = 0;
    addNum(12, 24, sum_int);
    std::cout << "sum_int = " << sum_int << std::endl;

    std::cout << "------------ template Type == float ------------\n";
    float sum_float = 0.f;
    addNum(12.1f, 2.4f, sum_float);
    std::cout << "sum_float = " << sum_float << std::endl;

    std::cout << "------------ template Type == double ------------\n";
    double sub_double = 0;
    subNum(23.4, 12.1, sub_double);
    std::cout << "sub_double = " << sub_double << std::endl;

    std::cout << "------------ template Type == int ------------\n";
    int mul = 0;
    mulNum(4, 12, mul);
    std::cout << "mul = " << mul << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\14_functionTemplate.cpp -std=c++23
// g++ .\14_functionTemplate.cpp -std=c++23
