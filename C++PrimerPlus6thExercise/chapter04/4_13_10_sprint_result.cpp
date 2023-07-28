/**
 * @file 4_13_10_sprint_result.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <cstddef>
#include <iostream>
#include <numeric>

/**
 * @brief 编写C++程序, 要求用户输入3次50米的跑步成绩,
 * 然后显示次数和平均成绩, 请用 std::array 对象存储数据
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const size_t num_repeat = 3;

    std::array<float, num_repeat> results;
    for (size_t i = 0; i < num_repeat; ++i)
    {
        std::cout << "Please enter the result of Sprint(50m): ";
        std::cin >> results[i];
    }

    // std::cout << "Please enter the first result of Sprint(50m): ";
    // std::cin >> results[0];

    // std::cout << "Please enter the second result of Sprint(50m): ";
    // std::cin >> results[1];

    // std::cout << "Please enter the third result of Sprint(50m): ";
    // std::cin >> results[2];

    float average = std::accumulate(results.cbegin(), results.cend(), 0.f) / num_repeat;

    std::cout << "The average result of " << num_repeat << " times Sprint: ";
    std::cout << average << std::endl;

    return 0;
}