/**
 * @file 10_10_4_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "10_10_4_sale_class.hpp"

#include <vector>


/**
 * @brief 编写C++程序, 理解C++中的命名空间
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    SALES::Sales sale_obj1;
    sale_obj1.showSales();

    std::vector<double> arr = {1.2, 3.4, 7.8, 9.0};
    SALES::Sales sale_obj2{arr};
    sale_obj2.showSales();

    SALES::Sales sale_obj3{sale_obj2};
    sale_obj3.append_value(66.6);
    sale_obj3.showSales();
    sale_obj3.append_value(88.8);
    sale_obj3.showSales();

    return 0;
}