/* exercise 7-2
** 练习7.2: 使用 2.6.2 节练习中编写了一个 Sales_data 类
** 请向这个类添加 combine 和 isbn 成员
** solution: 
**
*/

#ifndef EXERCISE7_2_H
#define EXERCISE7_2_H

#include <string>

struct Sales_data {
    std::string isbn() const { return bookNo; };
    Sales_data& combine(const Sales_data&);
    
    std::string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;
};

Sales_data& Sales_data::combine(const Sales_data& rhs)
{
    units_sold += rhs.units_sold;
    revenue += rhs.revenue;
    return *this;
}

#endif  // EXERCISE7_2_H