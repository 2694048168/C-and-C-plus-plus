/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

// #include "add.hpp"
#include <add.hpp>
#include <sub.hpp>

int main(int argc, const char** argv)
{
    std::cout << "Hello C++ world\n";

    std::cout << "the sum of two integer: " << add(42, 24) << "\n";

    std::cout << "the sub of two integer: " << sub(42, 24) << "\n";

    return 0;
}
