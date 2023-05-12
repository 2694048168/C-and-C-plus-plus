/**
 * @file param_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the parameters pass by pointer in function
 * @attention
 *
 */

#include <iostream>
#include <cfloat>

int foo_value(int x)
{
    x += 10;
    return x;
}

int foo_pointer(int *p)
{
    (*p) += 10;
    return *p;
}

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    int num1 = 20;
    int num2 = foo_value(num1);
    std::cout << "num1=" << num1 << std::endl;
    std::cout << "num2=" << num2 << std::endl;

    int *p = &num1;
    int num3 = foo_pointer(p);
    std::cout << "num1=" << num1 << std::endl;
    std::cout << "*p=" << *p << std::endl;
    std::cout << "num3=" << num3 << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ param_pointer.cpp
 * $ clang++ param_pointer.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */