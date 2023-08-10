/**
 * @file 11_9_4_time_class.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __TIME_CLASS_HPP__
#define __TIME_CLASS_HPP__

#include <iostream>

class Time
{
private:
    int hours;
    int minutes;

public:
    Time();
    Time(int h, int m = 0);

    void AddMin(int m);
    void AddHr(int h);

    void Reset(int h = 0, int m = 0);

    // operator '+' overloading
    friend Time operator+(const Time &t1, const Time &t);
    friend Time operator-(const Time &t1, const Time &t);
    friend Time operator*(const Time &t, double n);
    friend Time operator*(double m, const Time &t);

    friend std::ostream &operator<<(std::ostream &os, const Time &t);

    void Show() const;
};

#endif // !__TIME_CLASS_HPP__