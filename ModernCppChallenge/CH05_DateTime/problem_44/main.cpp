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
#include <iomanip>
#include <iostream>

/**
 * @brief Monthly calendar
 * 
 * Write a function that, given a year and month, 
 * prints to the console the month calendar
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
unsigned int week_day(const int y, const unsigned int m, const unsigned int d)
{
    if (m < 1 || m > 12 || d < 1 || d > 31)
        return 0;

    const auto dt   = std::chrono::year_month_day{std::chrono::year{y}, std::chrono::month{m}, std::chrono::day{d}};
    const auto tiso = std::chrono::weekday{dt};

    return (unsigned int)tiso.c_encoding();
}

void print_month_calendar(const int y, unsigned int m)
{
    std::cout << "Mon Tue Wed Thu Fri Sat Sun" << std::endl;

    auto first_day_weekday = week_day(y, m, 1);
    auto last_day          = (unsigned int)std::chrono::year_month_day_last(std::chrono::year{y}, std::chrono::month_day_last{std::chrono::month{m}}).day();

    unsigned int index = 1;
    for (unsigned int day = 1; day < first_day_weekday; ++day, ++index)
    {
        std::cout << "    ";
    }

    for (unsigned int day = 1; day <= last_day; ++day)
    {
        std::cout << std::right << std::setfill(' ') << std::setw(3) << day << ' ';
        if (index++ % 7 == 0)
            std::cout << std::endl;
    }

    std::cout << std::endl;
}

// ------------------------------
int main(int argc, char **argv)
{
    unsigned int y = 0, m = 0;
    std::cout << "Year:";
    std::cin >> y;
    std::cout << "Month:";
    std::cin >> m;

    print_month_calendar(y, m);

    return 0;
}
