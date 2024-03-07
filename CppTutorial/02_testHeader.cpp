/**
 * @file 02_testHeader.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 中头文件的使用方式
 * @version 0.1
 * @date 2024-03-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "02_headerFile.hpp"

#include <iostream>

// ===================================
int main(int argc, const char **argv)
{
    std::cout << "============ 测试头文件使用方式 ============\n";

    int value1 = 42;
    int value2 = 24;

    std::cout << addNumber(value1, value2) << std::endl;
    std::cout << subNumber(value1, value2) << std::endl;
    std::cout << mulNumber(value1, value2) << std::endl;
    std::cout << divNumber(value1, value2) << std::endl;

    std::cout << "==========================================\n";

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\02_testHeader.cpp 02_headerFile.cpp -std=c++23
// g++ .\02_testHeader.cpp 02_headerFiler.cpp -std=c++23
