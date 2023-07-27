/**
 * @file 4_13_6_structure_array.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>
#include <vector>

struct CandyBar
{
    std::string  brand_name;
    float        weight;
    unsigned int calories;
};

inline void print_info(const CandyBar &candy)
{
    std::cout << "The candy name is " << candy.brand_name << ",\n";
    std::cout << "and the candy weight is " << candy.weight << ",\n";
    std::cout << "and the candy calories is " << candy.calories << ".\n";
}

/**
 * @brief 声明指定的结构体, 并显示该结构体变量的信息
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // CandyBar snack = {"Mocha Munch", 2.3, 350};
    const unsigned num_candy = 3;

    CandyBar snacks[num_candy];
    for (size_t i = 0; i < num_candy; ++i)
    {
        snacks[i] = {"Mocha Munch", 2.3, 350};
    }

    for (size_t i = 0; i < num_candy; ++i)
    {
        print_info(snacks[i]);
        std::cout << "---------------------------" << std::endl;
    }

    // --------------- vector --------------------
    std::cout << std::endl;

    std::vector<CandyBar> candy_vec;
    for (size_t i = 0; i < num_candy; ++i)
    {
        candy_vec.push_back({"Mocha Munch", 2.3, 350});
    }

    for (const auto elem : candy_vec)
    {
        print_info(elem);
        std::cout << "---------------------------" << std::endl;
    }

    return 0;
}