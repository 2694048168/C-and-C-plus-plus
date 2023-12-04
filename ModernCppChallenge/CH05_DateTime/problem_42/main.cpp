/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-12-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <iostream>

/**
 * @brief Day and week of the year
 * 
 * Write a function that, given a date, 
 * returns the day of the year (from 1 to 365 or 366 for leap years)
 * and another function that, for the same input,
 * returns the calendar week of the year.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
unsigned int calendar_week(const int y, const unsigned int m, const unsigned int d)
{
    if (m < 1 || m > 12 || d < 1 || d > 31)
        return 0;

    const auto dt   = std::chrono::year_month_day{std::chrono::year{y}, std::chrono::month{m}, std::chrono::day{d}};
    const auto tiso = std::chrono::weekday{dt};

    return (unsigned int)tiso.c_encoding();
}

int day_of_year(const int y, const unsigned int m, const unsigned int d)
{
    if (m < 1 || m > 12 || d < 1 || d > 31)
        return 0;

    return (std::chrono::sys_days{std::chrono::year{y} / std::chrono::month{m} / std::chrono::day{d}} - std::chrono::sys_days{std::chrono::year{y} / std::chrono::January / 0}).count();
}

// ------------------------------
int main(int argc, char **argv)
{
    int          y = 0;
    unsigned int m = 0, d = 0;
    std::cout << "Year:";
    std::cin >> y;
    std::cout << "Month:";
    std::cin >> m;
    std::cout << "Day:";
    std::cin >> d;

    std::cout << "Calendar week:" << calendar_week(y, m, d) << std::endl;
    std::cout << "Day of year:" << day_of_year(y, m, d) << std::endl;

    return 0;
}
