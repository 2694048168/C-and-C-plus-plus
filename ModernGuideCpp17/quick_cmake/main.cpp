/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

#include "add.hpp"
#include "sub.hpp"
#include "mul.hpp"


int main(int argc, const char** argv) 
{
    // const int SIZE = 24;

    std::cout << "Hello C++ world\n";
    std::printf("Hello C++ world\n");

    int res = add(1, 2);
    std::cout << "1 + 2 = " << res << std::endl;

    res = sub(9, 1);
    std::cout << "9 - 1 = " << res << std::endl;

    res = mul(3, 5);
    std::cout << "3 * 5 = " << res << std::endl;

    return 0;
}
