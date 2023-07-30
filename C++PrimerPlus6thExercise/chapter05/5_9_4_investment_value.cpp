/**
 * @file 5_9_4_investment_value.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstddef>
#include <iostream>

/**
 * @brief 编写C++程序, 计算单利年投资(10%)和复利年投资(5%), 
 * 多少年后复利投资超过单利投资, 并同时显示两者投资价值
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // const unsigned long long int capital = 100;
    const float capital                = 100;
    const float simple_interest_rate   = 0.10f;
    const float compound_interest_rate = 0.05f;

    // 单纯为了计算速度, 采用 float, 否则应该为了追求精准使用 double
    float simple_interest   = 0.f;
    float compound_interest = 0.f;
    float current_capital   = capital;

    size_t num_year = 1;
    do
    {
        simple_interest += capital * simple_interest_rate;

        compound_interest += current_capital * compound_interest_rate;
        current_capital   = compound_interest + current_capital;

        ++num_year;
    }
    while (simple_interest >= compound_interest);

    std::cout << "After " << num_year;
    std::cout << ", the value of compound interest is more than simple interest.\n";
    std::cout << "the value of simple interest is: " << simple_interest << "\n";
    std::cout << "the value of compound interest is: " << compound_interest << "\n";

    return 0;
}