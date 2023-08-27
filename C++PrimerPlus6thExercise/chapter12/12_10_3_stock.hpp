/**
 * @file 12_10_3_stock.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __STOCK_HPP__
#define __STOCK_HPP__

#include <string>

class Stock
{
    // private:
    //     std::string company;
    //     int         shares;
    //     double      share_val;
    //     double      total_val;

    //     void set_tot()
    //     {
    //         total_val = shares * share_val;
    //     }

    // public:
    //     Stock(); // default constructor
    //     Stock(const std::string &co, long n = 0, double pr = 0.0);
    //     ~Stock(); // do-nothing destructor

    //     void buy(long num, double price);
    //     void sell(long num, double price);
    //     void update(double price);

    //     // void show() const;
    //     friend std::ostream &operator<<(std::ostream &os, const Stock &st);

    //     const Stock &top_val(const Stock &s) const;

    // --------------------------------------------
private:
    char  *company;
    int    shares;
    double share_val;
    double total_val;

    void set_tot()
    {
        total_val = shares * share_val;
    }

public:
    Stock();
    Stock(const char *co, long n = 0, double pr = 0.0);
    Stock(const Stock &st);
    ~Stock();

    void buy(long num, double price);
    void sell(long num, double price);
    void update(double price);

    const Stock         &top_val(const Stock &s) const;
    friend std::ostream &operator<<(std::ostream &os, const Stock &st);
    // -----------------------------------------------------------------
};

#endif // !__STOCK_HPP__