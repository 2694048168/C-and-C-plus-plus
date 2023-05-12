/**
 * @file recursion.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the function recursion in C++
 * @attention
 *
 */

#include <iostream>

void mydiv2(float val)
{
    std::cout << "Entering val = " << val << std::endl;
    if (val > 1.0f)
    {
        mydiv2(val / 2); /* function call itself. */
    }
    else
    {
        std::cout << "-------------------------" << std::endl;
    }

    std::cout << "Leaving val = " << val << std::endl;
}


/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    // call the recursive function
    mydiv2(1024);

    mydiv2(1000);

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ recursion.cpp
 * $ clang++ recursion.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */