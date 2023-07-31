/**
 * @file 6_11_2_donation_statistics.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <ctype.h>

#include <array>
#include <iostream>

/**
 * @brief 编写C++程序, 统计捐赠的数量(直到非数字截止), 并计算其均值和超过均值的个数.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned int max_size = 12;

    std::cout << "Please enter the number of donations.\n";
    std::cout << "You must enter " << max_size << " donations.\n";

    std::array<double, max_size> donations;

    unsigned int idx = 0;
    for (; idx < max_size; ++idx)
    {
        std::cout << "number of donation #" << idx + 1 << ": ";
        // TODO 如何实现非数字结束用户输入?
        while (!(std::cin >> donations[idx]))
        {
            std::cin.clear();                        // reset input
            while (std::cin.get() != '\n') continue; // get rid of bad input
            std::cout << "Please enter a number: ";
        }
    }

    // calculate average
    double total = 0.0;
    for (idx = 0; idx < max_size; ++idx)
    {
        total += donations[idx];
    }
    double average = total / max_size;
    std::cout << "The average = " << average << " for donations\n";

    // greater than 'average',
    unsigned int count = 0;
    for (const auto elem : donations)
    {
        if (elem > average)
        {
            ++count;
        }
        else
        {
            continue;
        }
    }
    std::cout << "The number of greater than average is " << count << "\n";

    return 0;
}