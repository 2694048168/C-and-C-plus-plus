/**
 * @file 3_7_5_population_percentage.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 编写C++程序, 要求用户输入全球人口总数和一个国家的人口总数,
 * 并显示该国家人口数量站全球总人口的百分比.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter the world's population: ";
    unsigned long long global_total_population = 0;
    std::cin >> global_total_population;

    std::cout << "Please enter the China population: ";
    unsigned long long china_total_population = 0;
    std::cin >> china_total_population;

    // Attention: the integer division in C++,
    float percentage = (float)china_total_population / global_total_population;

    std::cout << "the population percentage is " << percentage * 100 << "%\n";

    return 0;
}