/**
 * @file constant_constexpr.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief const and constant expression 常量表达式; 编译器优化; 编译时和运行时; 
 * 查看源代码后的汇编代码; 查看可执行二进制的反汇编代码
 * @version 0.1
 * @date 2022-01-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>

#define LEN 10

int len_function()
{
    int i = 2;
    return i;
}

constexpr int len_function_constexpr()
{
    return 5;
}

constexpr int fibonacci(const int n)
{
    return n == 1 || n == 2 ? 1 : fibonacci(n - 1) + fibonacci(n - 2);
}

int main(int argc, char **argv)
{
    char arr_1[10];  /* legal */
    char arr_2[LEN]; /* legal */

    int len = 10;
    char arr_3[len]; /* illegal without compiler optimizations, want to reproduce the error, need to use old version compiler*/

    const int len_2 = len + 1;
    constexpr int len_2_constexpr = 1 + 2 + 3;
    char arr_4[len_2];           /* illegal, but ok for most of the compilers */
    char arr_5[len_2_constexpr]; /* legalf */

    char arr_6[len_function() + 5];           /* illegal */
    char arr_7[len_function_constexpr() + 1]; /* legal */

    std::cout << fibonacci(10) << std::endl;

    return 0;
}
