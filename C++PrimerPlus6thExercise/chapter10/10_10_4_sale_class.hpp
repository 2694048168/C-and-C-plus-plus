/**
 * @file 10_10_4_sale_class.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __SALE_CLASS_HPP__
#define __SALE_CLASS_HPP__

#include <vector>

namespace SALES {

class Sales
{
private:
    std::vector<double> sales;

    double average;
    double max;
    double min;

public:
    // constructor
    Sales() = default;
    Sales(const std::vector<double> vec);

    // copy constructor
    Sales(const Sales &s);

    void append_value(const double val);

    void showSales() const;
};

} // namespace SALES

#endif // !__SALE_CLASS_HPP__
