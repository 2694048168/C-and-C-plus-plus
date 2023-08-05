/**
 * @file 8_8_2_function_pass.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

struct CandyBar
{
    const char  *brand_name;
    double       weight;
    unsigned int calories;
};

void init_setter(CandyBar &candy, const char *name = "Millennium Munch", const double weight = 2.85,
                 const unsigned calory = 350)
{
    candy.brand_name = name;
    candy.weight     = weight;
    candy.calories   = calory;
}

void print_getter(const CandyBar &candy)
{
    std::cout << "The information of CandyBar: \n";
    std::cout << candy.brand_name << " ";
    std::cout << candy.weight << " ";
    std::cout << candy.calories << "\n";
    std::cout << "----------------------------- \n";
}

/**
 * @brief 编写C++程序, 针对结构体进行赋值初始化(构造函数)和显示函数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    CandyBar candy;
    init_setter(candy);

    print_getter(candy);

    CandyBar candy_new;
    init_setter(candy_new, "wei li", 42.24, 256);

    print_getter(candy_new);

    return 0;
}