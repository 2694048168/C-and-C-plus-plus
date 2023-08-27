/**
 * @file 12_10_3_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "12_10_3_stock.hpp"

#include <iostream>

const int SIZE = 4;

/**
 * @brief 编写C++程序，TODO 利用动态内存开辟和释放
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    {
        // create an array of initialized objects
        Stock stocks[SIZE] = {Stock("NanoSmart", 12, 20.0), Stock("Boffo Objects", 200, 2.0),
                              Stock("Monolithic Obelisks", 130, 3.25), Stock("Fleep Enterprises", 60, 6.5)};

        std::cout << "Stock holdings:\n";
        int st;
        for (st = 0; st < SIZE; st++)
        {
            std::cout << stocks[st];
        }

        // set pointer to first element
        const Stock *top = &stocks[0];
        for (st = 1; st < SIZE; st++) top = &top->top_val(stocks[st]);

        // now top points to the most valuable holding
        std::cout << "\nMost valuable holding:\n";
        std::cout << *top;
    }

    return 0;
}