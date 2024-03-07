/**
 * @file 00_cppEnvironment.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 快速搭建学习 C++ 的编程环境
 * @version 0.1
 * @date 2024-03-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// ===================================
int main(int argc, const char **argv)
{
    std::string language = "C++";
    std::cout << "Welcome to the ";
    std::cout << language << " programming world\n";

    std::cout << "The C++ standard: " << __cplusplus << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\00_cppEnvironment.cpp -std=c++23
// g++ .\00_cppEnvironment.cpp -std=c++23
