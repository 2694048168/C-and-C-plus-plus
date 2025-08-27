/**
 * @file Math_Impl.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

module MathModule;

namespace MathModule {

int add(int a, int b)
{
    return a + b;
}

int sub(int a, int b)
{
    return a - b;
}

int div(int a, int b)
{
    return a / b;
}

int mul(int a, int b)
{
    return a * b;
}

MathDemo::MathDemo() {}

MathDemo::~MathDemo() {}

int MathDemo::Add(int a, int b)
{
    return a + b;
}

int MathDemo::Sub(int a, int b)
{
    return a - b;
}

int MathDemo::Div(int a, int b)
{
    return a / b;
}

int MathDemo::Mul(int a, int b)
{
    return a * b;
}

} // namespace MathModule
