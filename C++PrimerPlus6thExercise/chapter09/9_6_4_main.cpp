/**
 * @file 9_6_4_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "9_6_4_namespace.hpp"

#include <iostream>

/**
 * @brief 编写C++程序, 理解C++中的命名空间
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    SALES::Sales sale_obj;

    double arr[SALES::QUARTERS] = {1.2, 3.4, 7.8, 9.0};

    SALES::setSales(sale_obj, arr, SALES::QUARTERS);
    SALES::showSales(sale_obj);

    // --------------------
    SALES::Sales sale;

    SALES::setSales(sale);
    SALES::showSales(sale);

    return 0;
}