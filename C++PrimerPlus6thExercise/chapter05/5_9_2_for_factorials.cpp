/**
 * @file 5_9_2_for_factorials.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <iostream>
// #include <stddef.h>
#include <cstddef>

/**
 * @brief 编写C++程序, 利用 for 循环完成阶乘的计算
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // Original source code,
    const unsigned int array_size = 16;
    long long          factorials[array_size];

    factorials[1] = factorials[0] = 1LL;
    for (int i = 2; i < array_size; i++)
    {
        factorials[i] = i * factorials[i - 1];
    }

    for (int i = 0; i < array_size; i++)
    {
        std::cout << i << "! = " << factorials[i] << std::endl;
    }

    //---------------------------------
    std::cout << "\n-------------------------------" << std::endl;
    const size_t size_array = 101;

    std::array<long double, size_array> factorial_array;
    factorial_array[1] = factorial_array[0] = 1L;
    for (int i = 2; i < size_array; i++)
    {
        factorial_array[i] = i * factorial_array[i - 1];
    }

    for (int i = 0; i < size_array; i++)
    {
        std::cout << i << "! = " << factorial_array[i] << std::endl;
    }
    
    return 0;
}