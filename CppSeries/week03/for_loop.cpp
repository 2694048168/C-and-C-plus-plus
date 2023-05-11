/**
 * @file for_loop.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief for loop and for-range in C++11.
 * @attention while and for loop can convert.
 *
 */

#include <iostream>
#include <string>
#include <cstdlib> /* EXIT_SUCCESS marco */

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    int sum = 0;
    for (size_t i = 0; i < 100; i++)
    {
        sum += i;
    }
    std::cout << "the result of sum: " << sum << std::endl;

    const std::string msg = "Hello CPP World.";
    for (auto &ch : msg)
    {
        std::cout << ch;
    }
    std::cout << std::endl;

    const char *msg_ptr = "Hello CPP World.";
    /* invalid range expression of type 'const char';
    no viable 'begin' function available.
    ---------------------------------------- */
    // for (auto &ch : msg_ptr) /* error */
    // for (auto &ch : (*msg_ptr)) /* error */
    for (auto &ch : std::string(msg_ptr))
    {
        std::cout << ch;
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ for_loop.cpp
 * $ clang++ for_loop.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */