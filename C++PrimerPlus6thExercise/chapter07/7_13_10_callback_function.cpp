/**
 * @file 7_13_10_callback_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

inline double calculate(const double &x, const double &y, double (*func)(const double &, const double &))
{
    return func(x, y);
}

inline double add(const double &x, const double &y)
{
    return x + y;
}

inline double multiple(const double &x, const double &y)
{
    return x * y;
}

/**
 * @brief 编写C++程序, 利用函数指针构造回调函数思想
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    double result = calculate(2.5, 10.4, add);
    std::cout << "The result of add calculate: " << result << "\n";

    result = calculate(2.5, 10.4, multiple);
    std::cout << "The result of add calculate: " << result << "\n";

    // 函数指针, 本质是一个指针, 一个指向函数地址的指针;
    // 函数指针数组, 本质是一个数组, 一个存储函数指针的数组;
    std::cout << "===========================================\n";
    const unsigned int size = 6;
    double (*pf[size])(double, double);
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << "The result of add calculate: ";
        std::cout << calculate(i + 1, i + 2, add) << "\n";
    }
    std::cout << "===========================================\n";

    unsigned int idx = 1;
    // TODO 循环退出的条件?
    while (std::cin)
    {
        double value1;
        double value2;

        std::cout << "Enter value #" << idx << ": ";
        std::cin >> value1;

        ++idx;

        std::cout << "Enter value #" << idx << ": ";
        std::cin >> value2;

        double result = calculate(value1, value2, add);
        std::cout << "The result of add calculate: " << result << "\n";
    }

    return 0;
}