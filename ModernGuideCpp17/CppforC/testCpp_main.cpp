/**
 * @file testCpp_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// extern "C"
// {
// #include "libmathC/custom_math.h"
// }

#include "libmathC/custom_math.h"
#include <iostream>

// -------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "====================================";
    std::cout << "\nThe sum is : " << 24 << " + " << 12 << " = " << custom_add(24, 12);
    std::cout << "\nThe sub is : " << 24 << " - " << 12 << " = " << custom_sub(24, 12);
    std::cout << "\nThe mul is : " << 24 << " * " << 12 << " = " << custom_mul(24, 12);
    std::cout << "\nThe div is : " << 24 << " / " << 12 << " = " << custom_div(24, 12);
    std::cout << "\n====================================\n";

    return 0;
}
