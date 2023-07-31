/**
 * @file 6_11_5_tax_code.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cctype>
#include <iostream>

/**
 * @brief 编写C++程序, 根据用户的输入的收入, 按照梯度范式计算所收的税
 * First 0~5,000       : 0% tax
 * Next  5000~15,000   : 10% tax
 * Next  15000~35,000  : 15% tax
 * after 35,000~       : 20% tax
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // double income = 0;
    // 为了计算效率, 使用 float; 为了更加精准的结果, 应该采用 double;
    float income = 0.f;
    // float first_tax_rate = 0.f;
    float second_tax_rate = 0.1f;
    float third_tax_rate  = 0.15f;
    float fourth_tax_rate = 0.2f;

    std::cout << "Please enter your income: ";
    std::cin >> income;
    // while (isdigit(income) && income >= 0)
    // isdigit function range(0-255)? how the solution of this problem?
    while (income >= 0)
    {
        float total_tax = 0.f;

        if (income <= 5000)
        {
            total_tax = 0.f;
        }
        else if (income > 5000 && income <= 15000)
        {
            total_tax = (income - 5000) * second_tax_rate;
        }
        else if (income > 15000 && income <= 35000)
        {
            total_tax = (income - 15000) * third_tax_rate;
            total_tax += 10000 * second_tax_rate;
        }
        else
        {
            total_tax = (income - 35000) * fourth_tax_rate;
            total_tax += 20000 * third_tax_rate;
            total_tax += 10000 * second_tax_rate;
        }

        std::cout << "Your total tax should be: " << total_tax << std::endl;

        std::cout << "Please enter your income: ";
        std::cin >> income;
    }

    return 0;
}