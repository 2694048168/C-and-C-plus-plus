#pragma once

#include <iostream>
#include <string>

class MyTime
{
private:
    int hours;
    int minutes;

public:
    MyTime() : hours(0), minutes(0) {}
    MyTime(int h, int m) : hours(h), minutes(m) {}

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
        sum.hours +=  sum.minutes / 60;
        sum.minutes %= 60;

        return sum;
    }
    // operator(+= INT) overloading
    MyTime & operator+=(int m) 
    {
        this->minutes += m;
        this->hours +=  this->minutes / 60;
        this->minutes %= 60;

        return *this;
    }
    // operator(+ STRING) overloading
    MyTime operator+(const std::string str) const
    {
        MyTime sum = *this;
        if(str=="one hour")
            sum.hours = this->hours + 1;
        else
            std::cerr<< "Only \"one hour\" is supported." << std::endl;
        return sum;
    }

    std::string getTime() const
    {
        return std::to_string(this->hours) + " hours and " + std::to_string(this->minutes) + " minutes.";
    }
};