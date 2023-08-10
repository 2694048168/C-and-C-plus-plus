/**
 * @file 11_9_6_stone_weight_overload.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __STONE_WEIGHT_OVERLOAD_HPP__
#define __STONE_WEIGHT_OVERLOAD_HPP__

#include <iostream>

class StoneWeight
{
public:
    enum STATUS
    {
        integer_pounds_form        = 0,
        floating_point_pounds_form = 1,
        NUM_STATUS                 = 2
    };

private:
    // pounds per stone
    static const int Lbs_per_stn = 14;

    STATUS status = integer_pounds_form;
    int    stone;    // whole stones
    double pds_left; // fractional pounds
    double pounds;   // entire weight in pounds

public:
    StoneWeight(double lbs);          // constructor for double pounds
    StoneWeight(int stn, double lbs); // constructor for stone, lbs
    StoneWeight();                    // default constructor
    ~StoneWeight();

    void   set_status(STATUS status);
    double get_pounds();

    StoneWeight operator+(const StoneWeight &s);
    StoneWeight operator-(const StoneWeight &s);
    StoneWeight operator*(double val);

    friend StoneWeight operator*(double val, const StoneWeight &s)
    {
        return val * s;
    } // inline implement

    // void show_lbs() const; // show weight in pounds format
    // void show_stn() const; // show weight in stone format
    friend std::ostream &operator<<(std::ostream &os, const StoneWeight &s);

    // 关系运算符重载
    bool operator<(const StoneWeight &s);
    bool operator<=(const StoneWeight &s);
    bool operator>(const StoneWeight &s);
    bool operator>=(const StoneWeight &s);

    friend bool operator==(const StoneWeight &s1, const StoneWeight &s);
    friend bool operator!=(const StoneWeight &s1, const StoneWeight &s);
};

#endif // !__STONE_WEIGHT_OVERLOAD_HPP__