/**
 * @file 9_6_4_namespace.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __NAMESPACE_HPP__
#define __NAMESPACE_HPP__

namespace SALES {

const int QUARTERS = 4;

struct Sales
{
    double sales[QUARTERS];
    double average;
    double max;
    double min;
};

// copies the lesser of 4 or n items from the array ar
// to the sales member of s and computes and stores the
// average, maximum, and minimum values of the entered items;
// remaining elements of sales, if any, set to 0
void setSales(Sales &s, const double ar[], int n);

// gathers sales for 4 quarters interactively, stores them
// in the sales member of s and computes and stores the
// average, maximum, and minimum values
void setSales(Sales &s);

// display all information in structure s
void showSales(const Sales &s);

} // namespace SALES

#endif // !__NAMESPACE_HPP__
