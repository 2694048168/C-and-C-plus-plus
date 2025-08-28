/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-28
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "Math/Math.h"

#include <iostream>

// -------------------------------------
int main(int argc, const char *argv[])
{
    const int a = 44;
    const int b = 2;

    Math::MathExample calculator;
    std::cout << "calculator Add = " << calculator.Add(a, b) << std::endl;
    std::cout << "calculator Sub = " << calculator.Sub(a, b) << std::endl;
    std::cout << "calculator Div = " << calculator.Div(a, b) << std::endl;
    std::cout << "calculator Mul = " << calculator.Mul(a, b) << std::endl;

    return 0;
}
