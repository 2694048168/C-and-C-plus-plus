/**
 * @file goto_statement.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the goto statement in C and C++.
 * @attention 使得程序逻辑不通畅(混乱), 指令执行
 *
 */

#include <iostream>
#include <cstdlib> /* EXIT_SUCCESS marco */


float mysquare(float value)
{
    if (value >= 1.0f || value <= 0)
    {
        std::cerr << "the input is out of range.\n";
        goto EXIT_ERROR;
    }
    return value * value;

    EXIT_ERROR:
        std::cout << "goto handle the error in function." << std::endl;
        return 0.0f;
}

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    float value;
    std::cout << "Input a floating-point number." << std::endl;
    std::cin >> value;

    float result = mysquare(value);

    if (result > 0)
        std::cout << "The square is " << result << "." << std::endl;

    EXIT_ERROR: /* 不小心多写一个 符号(goto) */
        std::cout << "goto handle the error in [main] function." << std::endl;
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ goto_statement.cpp
 * $ clang++ goto_statement.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */