/**
 * @file 4_13_5_structure_info.cpp
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

struct CandyBar
{
    std::string brand_name;
    float weight;
    unsigned int calories;
};

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
    CandyBar snack{"Mocha Munch", 2.3, 350};

    std::cout << "The candy name is " << snack.brand_name << ",\n";
    std::cout << "and the candy weight is " << snack.weight << ",\n";
    std::cout << "and the candy calories is " << snack.calories << ".\n";

    return 0;
}