/**
 * @file single_demo.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief simple string demo about operator overloading
 * @attention
 *
 */

#include <iostream>
#include <string>

/**
 * @brief main function and entry point
 */
int main(int argc, char const *argv[])
{
    std::string s("Hello ");
    s += "C";
    s.operator+=(" and CPP!");

    std::cout << s << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ single_demo.cpp
 * $ clang++ single_demo.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */