/**
 * @file overload.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the function overloadding in C++
 * @attention the function signature in C++ and C
 *
 */

#include <iostream>

int mysum(const int x, const int y)
{
    std::cout << "sum(int, int) is called.\n";
    return x + y;
}

float mysum(const float x, const float y)
{
    std::cout << "sum(float, float) is called.\n";
    return x + y;
}

double mysum(const double x, const double y)
{
    std::cout << "sum(double, double) is called.\n";
    return x + y;
}

/* functions that differ only in their return type cannot be overloaded */
// double mysum(const int x, const int y)
// {
//     std::cout << "sum(int, int) is called.\n";
//     return x + y;
// }

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    std::cout << "sum = " << mysum(1, 2) << std::endl;
    std::cout << "sum = " << mysum(1.1f, 2.2f) << std::endl;
    std::cout << "sum = " << mysum(1.1, 2.2) << std::endl;

    /* which function will be called?
    error: call to 'mysum' is ambiguous
    ------------------------------------ */
    // std::cout << "sum = " << mysum(1, 2.2) << std::endl;
    // std::cout << "sum = " << mysum(1.f, 2.2) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ overload.cpp
 * $ clang++ overload.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */