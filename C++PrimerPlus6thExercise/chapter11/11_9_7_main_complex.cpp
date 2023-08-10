/**
 * @file 11_9_7_main_complex.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "11_9_7_my_complex.hpp"

#include <iostream>

/**
 * @brief 编写C++程序, 测试自定义的 class my_complex
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    MyComplex a(3.0, 4.0); // initialize to (3,4i)

    MyComplex c;
    std::cout << "Enter a complex number (q to quit):\n";
    while (std::cin >> c)
    {
        std::cout << "c is " << c << '\n';
        std::cout << "complex conjugate is " << ~c << '\n';
        std::cout << "a is " << a << "\n";
        std::cout << "a + c is " << a + c << '\n';
        std::cout << "a - c is " << a - c << '\n';
        std::cout << "a * c is " << a * c << '\n';
        std::cout << "2 * c is " << 2 * c << '\n';
        std::cout << "c * 2 is " << c * 2 << '\n';
        std::cout << "Enter a complex number (q to quit):\n";
    }
    std::cout << "Done!\n";

    return 0;
}