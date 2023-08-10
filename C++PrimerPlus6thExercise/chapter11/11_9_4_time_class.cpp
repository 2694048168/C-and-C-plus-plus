#include "11_9_4_time_class.hpp"

#include <iostream>

Time::Time()
{
    hours   = 0;
    minutes = 0;
}

Time::Time(int h, int m)
{
    hours   = h;
    minutes = m;
}

void Time::AddMin(int m)
{
    minutes += m;
    hours += minutes / 60;
    minutes = minutes % 60;
}

void Time::AddHr(int h)
{
    hours += h;
}

void Time::Reset(int h, int m)
{
    hours   = h;
    minutes = m;
}

// operator '+' overloading
Time operator+(const Time &t1, const Time &t)
{
    Time sum;
    sum.minutes = t1.minutes + t.minutes;
    sum.hours   = t1.hours + t.hours + sum.minutes / 60;
    sum.minutes = sum.minutes % 60;

    return sum;
}

Time operator-(const Time &t1, const Time &t)
{
    Time diff;
    int  tot1, tot2;
    tot1         = t.minutes + 60 * t.hours;
    tot2         = t1.minutes + 60 * t1.hours;
    diff.minutes = (tot2 - tot1) % 60;
    diff.hours   = (tot2 - tot1) / 60;
    return diff;
}

Time operator*(const Time &t, double mult)
{
    Time result;
    long total_minutes = t.hours * mult * 60 + t.minutes * mult;
    result.hours       = total_minutes / 60;
    result.minutes     = total_minutes % 60;
    return result;
}

Time operator*(double m, const Time &t)
{
    return t * m;
} // inline definition

std::ostream &operator<<(std::ostream &os, const Time &t)
{
    os << t.hours << " hours, " << t.minutes << " minutes";
    return os;
}

void Time::Show() const
{
    std::cout << "Now Time: " << hours << " hours, ";
    std::cout << minutes << " minutes" << std::endl;
    std::cout << "---------------------------------\n";
}