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
 * @brief Day of the week
 * 
 * Write a function that, given a date, determines the day of the week. 
 * This function should return a value between 1 (for Monday) and 7 (for Sunday).
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
unsigned int week_day(const int y, const unsigned int m, const unsigned int d)
{
    using namespace std::chrono;

    if (m < 1 || m > 12 || d < 1 || d > 31)
        return 0;

    const auto dt   = std::chrono::year_month_day{year{y}, month{m}, day{d}};
    const auto tiso = std::chrono::year_month_weekday{dt};

    return (unsigned int)tiso.weekday().c_encoding();
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

    std::cout << "Day of week:" << week_day(y, m, d) << std::endl;

    return 0;
}
