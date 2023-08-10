/**
 * @file 11_9_6_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "11_9_6_stone_weight_overload.hpp"

#include <algorithm>
#include <array>
#include <vector>

/**
 * @brief 编写C++程序, 对所有关系运算符进行重载
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned int SIZE_ARRAY = 6;

    std::array<StoneWeight, SIZE_ARRAY> stone_array{23.3, 32.9, 67.2};

    double input;
    for (size_t i = 3; i < SIZE_ARRAY; ++i)
    {
        std::cout << "Please enter the init value of #" << i + 1 << ": ";
        std::cin >> input;
        stone_array[i] = input;
    }

    std::vector<double> weight_vec;

    unsigned int count = 0;

    StoneWeight standard_stone{11};
    for (size_t i = 0; i < SIZE_ARRAY; ++i)
    {
        weight_vec.push_back(stone_array[i].get_pounds());
        if (stone_array[i] >= standard_stone)
        {
            ++count;
        }
    }

    std::cout << "The number for pounds of Stone is more than 11: ";
    std::cout << count << "\n";

    std::cout << "The value of max stone: " << *std::max_element(weight_vec.cbegin(), weight_vec.cend()) << "\n";

    std::cout << "The value of min stone: " << *std::min_element(weight_vec.cbegin(), weight_vec.cend()) << "\n";

    return 0;
}