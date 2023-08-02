/**
 * @file 7_13_4_simple_lottery.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iomanip> /* std::setprecision */
#include <iostream>

void probability(const unsigned &numbers, const unsigned &picks, double &prob)
{
    double   n;
    unsigned p;

    // 如何计算排列组合的概率：C_5^2 =
    for (n = numbers, p = picks; p > 0; n--, p--)
    {
        prob = prob * n / p;
    }
}

/**
 * @brief 编写C++程序, 计算简单彩票的头奖的几率
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // double total;
    // double choices;
    // std::cout << "Enter the total number of choices on the game card and\n"
    //              "the number of picks allowed:\n";

    // while ((std::cin >> total >> choices) && choices <= total)
    // {
    //     std::cout << "You have one chance in ";
    //     std::cout << probability(total, choices); // compute the odds
    //     std::cout << " of winning.\n";
    //     std::cout << "Next two numbers (q to quit): ";
    // }
    // std::cout << "bye\n";

    // field number [1~47] to select 5 number;
    double prob_field = 1.0;
    probability(47, 5, prob_field);

    //  special number [1~27] to select 1 number;
    double prob_special = 1.0;
    probability(27, 1, prob_special);

    double prob = 1.0 / (prob_field * prob_special);
    std::cout << "The probability of winning is: " << prob << std::endl;
    std::cout << "The probability of winning is: " << std::setprecision(12) << prob << std::endl;

    return 0;
}