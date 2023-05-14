/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Default constructor | Default copy constructor 
 *        | Default copy assignment 
 * @attention How pointer members work by default
 *
 */

#include <iostream>

#include "mystring.hpp"

/**
 * @brief main function
*/
int main(int argc, char const *argv[])
{
    MyString str1(10, "Shenzhen");
    std::cout << "str1: " << str1 << std::endl;

    MyString str2 = str1;
    std::cout << "str2: " << str2 << std::endl;

    MyString str3;
    std::cout << "str3: " << str3 << std::endl;
    str3 = str1;
    std::cout << "str3: " << str3 << std::endl;

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