/**
 * @file 00_constexpr.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief constexpr 关键字指示编译器在编译时期对表达式求值(如果可能得话)
 * 函数计算参数 number 的平方根, 从1开始, 该函数的递增局部变量 val,
 * 直到 val * val ≥ number;
 * 如果 val * val == number, 则 返回 val; 否则返回 val - 1;
 * 
 * @param number 
 * @return constexpr int 
 * 
 * @note 该函数的调用已有一个字面值, 所以编译器理论上可以直接计算表达式结果, 结果将只有一个值. 
 * @note https://godbolt.org/z/Wra1rj51r
 * @note <<The Art of Assembly Language>> second-edit by Randall Hyde
 * @note <<Professional Assembly Language>> by Richard Blum
 */
constexpr int iSqrt(int number)
{
    int val = 1;
    while (val * val < number)
    {
        ++val;
    }
    return val - (val * val != number);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    constexpr int x = iSqrt(1764);

    printf("The value of x = %d\n", x);

    return 0;
}
