#pragma once

#include <iostream>
#include <string>

class MyTime
{
private:
    int hours;
    int minutes;

public:
    MyTime() : hours(0), minutes(0)
    {
        std::cout << "Constructor MyTime()" << std::endl;
    }
    MyTime(int m) : hours(0), minutes(m)
    {
        std::cout << "Constructor MyTime(int)" << std::endl;
        this->hours += this->minutes / 60;
        this->minutes %= this->minutes;
    }
    MyTime(int h, int m) : hours(h), minutes(m)
    {
        std::cout << "Constructor MyTime(int, int)" << std::endl;
    }

    // prefix increment
    MyTime& operator++()
    {
        this->minutes++;
        this->hours += this->minutes / 60;
        this->minutes = this->minutes % 60;

        return *this;
    }
    // postfix increment
    MyTime operator++(int)
    {
        // keep the old value, why not postfix!
        MyTime old_temp_var = *this;

        operator++(); /* prefix incremnet */

        return old_temp_var;
    }

    // implicit conversion(整数表示时间的分钟数)
    operator int() const
    {
        std::cout << "operator int()" << std::endl;
        return this->hours * 60 + this->minutes;
    }
    // explicit conversion
    explicit operator float()
    {
        std::cout << "explicit operator float()" << std::endl;
        return float(this->hours * 60 + this->minutes);
    }
    
    // operator(= assignment) overloading
    MyTime & operator=(int m)
    {
        std::cout << "operator=(int)" << std::endl;
        this->hours = 0;
        this->minutes = m;
        this->hours =  this->minutes / 60;
        this->minutes %= 60;
        
        return *this;
    }

    // operator(+) overloading
    MyTime operator+(const MyTime &t) const
    {
        MyTime sum;
        sum.minutes = this->minutes + t.minutes;
        sum.hours = this->hours + t.hours;

        sum.hours += sum.minutes / 60;
        sum.minutes %= 60;

        return sum;
    }
    // operator(+=) overloading
    MyTime &operator+=(const MyTime &t)
    {
        this->minutes += t.minutes;
        this->hours += t.hours;

        this->hours += this->minutes / 60;
        this->minutes %= 60;

        return *this;
    }

    // operator(+ INT) overloading
    MyTime operator+(int m) const
    {
        MyTime sum;
        sum.minutes = this->minutes + m;
        sum.hours = this->hours;
        sum.hours += sum.minutes / 60;
        sum.minutes %= 60;

        return sum;
    }
    // friend function for operator(INT +) overloading
    friend MyTime operator+(int m, const MyTime &t)
    {
        return t + m;
    }
    // operator(+= INT) overloading
    MyTime &operator+=(int m)
    {
        this->minutes += m;
        this->hours += this->minutes / 60;
        this->minutes %= 60;

        return *this;
    }
    // operator(+ STRING) overloading
    MyTime operator+(const std::string str) const
    {
        MyTime sum = *this;
        if (str == "one hour")
            sum.hours = this->hours + 1;
        else
            std::cerr << "Only \"one hour\" is supported." << std::endl;
        return sum;
    }

    std::string getTime() const
    {
        return std::to_string(this->hours) + " hours and " + std::to_string(this->minutes) + " minutes.";
    }

    // friend function for operator(<<) overloading
    friend std::ostream &operator<<(std::ostream &os, const MyTime &t)
    {
        std::string str = std::to_string(t.hours) + " hours and " + std::to_string(t.minutes) + " minutes.";
        os << str;

        return os;
    }
    // friend function for operator(>>) overloading
    friend std::istream &operator>>(std::istream &is, MyTime &t)
    {
        is >> t.hours >> t.minutes;
        t.hours += t.minutes / 60;
        t.minutes %= 60;

        return is;
    }
};