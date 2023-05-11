/**
 * @file exercises.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @brief the integerts range with unsigned and signed in C++.
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>

int myadd(int, int);
unsigned int myadd(unsigned int, unsigned int);

/**
 * @brief main function and the entry of program.
 */
int main(int argc, char const *argv[])
{
    std::cout << "the signed int :" << sizeof(int) << " bytes.\n";
    std::cout << "the signed int :" << 8 * sizeof(int) << " bits.\n";
    // 2^32 - 1 = 2147483647(0x7FFFFFFF) 最高位是符号位
    // -2^32 - 1 = 2147483648(0x80000000) 最高位是符号位

    std::cout << "the unsigned int : " << sizeof(unsigned int) << " bytes.\n";
    std::cout << "the signed int :" << 8 * sizeof(unsigned int) << " bits.\n";

    // exercies in week01
    int arg_1 = 2147483647;
    int arg_2 = 1;
    std::cout << "the result of myadd: " << myadd(arg_1, arg_2) << std::endl;

    // Attention the overflow in C++!
    unsigned int argument_1 = 2147483647;
    unsigned int argument_2 = 1;
    std::cout << "the result of myadd: " << myadd(argument_1, argument_2)
              << std::endl;

    return 0;
}

int myadd(int param_1, int param_2)
{
    return param_1 + param_2;
}

unsigned int myadd(unsigned int param_1, unsigned int param_2)
{
    return param_1 + param_2;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ exercises.cpp
 * $ clang++ exercises.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */