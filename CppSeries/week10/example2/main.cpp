/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief operator overloading for class or object in C++
 * @attention
 *
 */

#include <iostream>

#include "mytime.hpp"

int main(int argc, char const *argv[])
{
    MyTime t1(2, 40);
    std::cout << (t1 + 30).getTime() << std::endl;

    t1 += 30; //operator
    t1.operator+=(30); //function

    std::cout << t1.getTime() << std::endl;

    std::cout << (t1 + "one hour").getTime() << std::endl;
    std::cout << (t1 + "two hour").getTime() << std::endl;

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