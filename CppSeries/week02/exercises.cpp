/**
 * @file exercises.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Data expresent as Integers and float-point in C++.
 * @attention overflow | precision | float-point
 *
 */

#include <iostream>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    int num1 = 1234567890;
    int num2 = 1234567890;
    int sum = num1 + num2; /* overflow */
    // unsigned int sum = num1 + num2;
    std::cout << "sum = " << sum << std::endl;
    std::cout << std::endl;

    float f1 = 1234567890.0f;
    float f2 = 1.0f;
    float fsum = f1 + f2; /* float-point number precision problem */
    std::cout << "fsum = " << fsum << std::endl;
    std::cout << "(fsum == f1) is " << (fsum == f1) << std::endl;
    std::cout << std::endl;

    float f = 0.1f;
    float sum10x = f + f + f + f + f + f + f + f + f + f;
    float mul10x = f * 10;

    std::cout << "sum10x = " << sum10x << std::endl;
    std::cout << "mul10x = " << mul10x << std::endl;
    // Attention the float-point in computer stored.
    std::cout << "(sum10x == 1) is " << (sum10x == 1.0) << std::endl;
    std::cout << "(mul10x == 1) is " << (mul10x == 1.0) << std::endl;
    std::cout << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ exercises.cpp
 * $ clang++ exercises.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */