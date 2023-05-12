/**
 * @file inline_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the inline function in C++
 * @attention macro and inline function
 *
 */

#include <iostream>

inline float max_function(float a, float b)
{
    // return ((a > b) ? a : b);
    if (a > b)
        return a;
    else
        return b;
}

// #define MAX_MACRO(a, b) a>b ? a : b
#define MAX_MACRO(a, b) (a) > (b) ? (a) : (b)

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. dynamic memory with malloc and free
    ----------------------------------------------- */
    int num1 = 42;
    int num2 = 24;
    int maxv = max_function(num1, num2);
    std::cout << maxv << std::endl;

    maxv = MAX_MACRO(num1, num2);
    std::cout << maxv << std::endl;

    maxv = MAX_MACRO(num1++, num2++);
    std::cout << maxv << std::endl;
    std::cout << "num1 = " << num1 << std::endl;
    std::cout << "num2 = " << num2 << std::endl;

    num1 = 0xAB09;
    num2 = 0xEF08;
    maxv = MAX_MACRO(num1 & 0xFF, num2 & 0xFF);
    std::cout << maxv << std::endl;

    /* Step 2. the reference in C++
    --------------------------------- */
    int num_value = 0;
    int & num_ref = num_value;
    std::cout << "num_value = " << num_value << std::endl;

    num_ref = 10;
    std::cout << "num_value = " << num_value << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ inline_function.cpp
 * $ clang++ inline_function.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */