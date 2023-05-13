/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief operator overloading and Automatic Conversions
 * @attention increment and decrement!
 *
 */

#include <iostream>

#include "mytime.hpp"

int main(int argc, char const *argv[])
{
    MyTime t1(1, 59);
    MyTime t2 = t1++;
    MyTime t3 = ++t1;

    std::cout << "t1 is " << t1 << std::endl;
    std::cout << "t2 is " << t2 << std::endl;
    std::cout << "t3 is " << t3 << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ main.cpp
 * $ clang++ main.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */