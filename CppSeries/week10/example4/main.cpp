/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief operator overloading and Automatic Conversions
 * @attention Automatic Conversions!
 *
 */

#include <iostream>

#include "mytime.hpp"

int main(int argc, char const *argv[])
{
    MyTime t1(1, 20);
    int minutes = t1;    /* implicit conversion */
    float f = float(t1); /* explicit conversion.  */
    std::cout << "minutes = " << minutes << std::endl;
    std::cout << "minutes = " << f << std::endl;

    MyTime t2 = 70;
    std::cout << "t2 is " << t2 << std::endl;

    MyTime t3;
    t3 = 80;
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