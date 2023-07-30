/**
 * @file 5_9_5_total_sales.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

/**
 * @brief 编写C++程序, 完成书籍<C++ for Fools>销售量的统计
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const std::vector<std::string> months{"January", "February", "March",     "April",   "May",      "June",
                                          "July",    "August",   "September", "October", "November", "December"};

    std::vector<int> month_sales;
    for (const auto elem : months)
    {
        std::cout << "Please enter the number sales of book for " << elem << ": ";
        unsigned int sales;
        std::cin >> sales;
        month_sales.push_back(sales);
    }

    unsigned total_sales = std::accumulate(month_sales.cbegin(), month_sales.cend(), 0);
    std::cout << "The total sales of book <C++ for Fools> in a year is: ";
    std::cout << total_sales << std::endl;

    return 0;
}