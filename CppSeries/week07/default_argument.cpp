/**
 * @file default_argument.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the default gargument for function
 * @attention
 *
 */

#include <iostream>
#include <cmath>

float norm(float x, float y, float z);
float norm(float x, float y, float z = 0);
float norm(float x, float y = 0, float z); /* why? */

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    std::cout << norm(3.0f) << std::endl;
    std::cout << norm(3.0f, 4.0f) << std::endl;
    std::cout << norm(3.0f, 4.0f, 5.0f) << std::endl;

    return 0;
}

float norm(float x, float y, float z)
{
    return std::sqrt(x * x + y * y + z * z);
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ default_argument.cpp
 * $ clang++ default_argument.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */