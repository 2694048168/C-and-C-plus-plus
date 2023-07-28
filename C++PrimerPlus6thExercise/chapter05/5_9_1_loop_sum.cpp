/**
 * @file 5_9_1_loop_sum.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 编写C++程序, 要求用户输入两个整数(从小到大), 
 * 计算包括该两个整数的范围之和
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter first small integer for a range: ";
    int small_integer = 0;
    std::cin >> small_integer;

    std::cout << "Please enter last large integer for a range: ";
    int large_integer = 0;
    std::cin >> large_integer;

    // TODO 如何保证用户第一次输入的是范围的起点(最小值), 程序的健壮性?

    int sum = 0;
    while (small_integer <= large_integer)
    {
        sum += small_integer;
        ++small_integer;
    }

    std::cout << "the sum of this range(from " << small_integer;
    std::cout << " to " << large_integer << ") is : " << sum << std::endl;

    return 0;
}