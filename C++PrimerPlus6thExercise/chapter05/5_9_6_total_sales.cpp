/**
 * @file 5_9_6_total_sales.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

/**
 * @brief 编写 C++ 程序, 完成书籍 <C++ for Fools> 三年的销售量的统计
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const size_t num_years  = 3;
    const size_t num_months = 12;

    const char *name_months[num_months] = {"January", "February", "March",     "April",   "May",      "June",
                                           "July",    "August",   "September", "October", "November", "December"};

    unsigned int sales_info[num_years][num_months];

    std::array<unsigned int, num_years> year_sale_list;

    for (size_t idx_year = 0; idx_year < num_years; ++idx_year)
    {
        for (size_t idx_month = 0; idx_month < num_months; ++idx_month)
        {
            std::cout << "Please enter the number sales of book for " << name_months[idx_month] << ": ";
            unsigned int sales;
            std::cin >> sales;
            sales_info[idx_year][idx_month] = sales;
        }

        unsigned year_sales = std::accumulate(&sales_info[idx_year][0], &sales_info[idx_year][num_months - 1], 0);

        year_sale_list[idx_year] = year_sales;

        std::cout << "The year sales of book <C++ for Fools> in " << idx_year + 1;
        std::cout << " year is: " << year_sales << "\n";
    }

    unsigned total_sales = std::accumulate(year_sale_list.cbegin(), year_sale_list.cend(), 0);
    std::cout << "The total sales of book <C++ for Fools> for " << num_years << " years is: ";
    std::cout << total_sales << std::endl;

    return 0;
}