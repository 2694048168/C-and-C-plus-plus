/* exercise 7-41
** 练习7.41: 使用委托构造函数重新编写你的Sales_data 类，
** 给每个构造函数体添加一条语句，令其一旦执行就打印一条信息。
** 用各种可能的方式分别创建Sales_data对象，认真研究每次输出的信息直到你确实理解了委托构造函数的执行顺序。
**
*/

#ifndef EXERCISE7_41_H
#define EXERCISE7_41_H


#include <string>
#include <iostream>

class Sales_data 
{
    friend std::istream &read(std::istream &is, Sales_data &item);
    friend std::ostream &print(std::ostream &os, const Sales_data &item);
    friend Sales_data add(const Sales_data &lhs, const Sales_data &rhs);

public:
    Sales_data(const std::string &s, unsigned n, double p):bookNo(s), units_sold(n), revenue(n*p)
    { std::cout << "Sales_data(const std::string&, unsigned, double)" << std::endl; }
    
    Sales_data() : Sales_data("", 0, 0.0f)
    { std::cout << "Sales_data()" << std::endl; }
    
    Sales_data(const std::string &s) : Sales_data(s, 0, 0.0f)
    { std::cout << "Sales_data(const std::string&)" << std::endl; }
    
    Sales_data(std::istream &is);

    std::string isbn() const { return bookNo; }
    Sales_data& combine(const Sales_data&);
    
private:
    inline double avg_price() const;        

private:
    std::string bookNo;
    unsigned units_sold = 0;
    double revenue = 0.0;
};

inline double Sales_data::avg_price() const
{
    return units_sold ? revenue/units_sold : 0;
}

// declarations for nonmember parts of the Sales_data interface.
std::istream &read(std::istream &is, Sales_data &item);
std::ostream &print(std::ostream &os, const Sales_data &item);
Sales_data add(const Sales_data &lhs, const Sales_data &rhs);

#endif // EXERCISE7_41_H