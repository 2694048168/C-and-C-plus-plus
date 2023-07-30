/**
 * @file 5_9_3_accumulative_sum.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <numeric>

/**
 * @brief 编写C++程序, 要求用户输入一个整数数字,
 * 累计计算并报告用户输入的累计和, 直到用户输入为 0
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    long long sum = 0;
    long long value_integer;

    do
    {
        std::cout << "Please enter an integer of number: ";
        std::cin >> value_integer;

        sum += value_integer;
        std::cout << "The accumulative sum currently is: " << sum << std::endl;
    }
    while (value_integer);

    return 0;
}