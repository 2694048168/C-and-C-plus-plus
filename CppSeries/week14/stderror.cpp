/**
 * @file stderror.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Standard Output Stream and Standard Error Stream
 * @attention pipe: stdout(1) stdin(0) stderror(2)
 *
 */

#include <iostream>

void div2(int n)
{
    if (n % 2 != 0)
    {
        std::cerr << "[Error]: The input must be an even number. Here it's "
                  << n << ".\n";
    }
    else
    {
        int result = n / 2;
        std::cout << "[Info]: The result is " << result << ".\n";
    }
    return;
}

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    for (int n = -5; n <= 5; n++)
        div2(n);

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ stderror.cpp
 * $ clang++ stderror.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */