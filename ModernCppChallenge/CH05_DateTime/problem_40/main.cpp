/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <iostream>

/**
 * @brief Number of days between two dates
 * 
 * Write a function that, given two dates, 
 * returns the number of days between the two dates.
 * The function should work regardless of the order of the input dates.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
inline int number_of_days(const int y1, const unsigned int m1, const unsigned int d1, const int y2,
                          const unsigned int m2, const unsigned int d2)
{
    using namespace std::chrono;

    return (sys_days{year{y1} / month{m1} / day{d1}} - sys_days{year{y2} / month{m2} / day{d2}}).count();
}

inline int number_of_days(const std::chrono::sys_days &first, const std::chrono::sys_days &last)
{
    return (last - first).count();
}

// ------------------------------
int main(int argc, char **argv)
{
    unsigned int y1 = 0, m1 = 0, d1 = 0;
    std::cout << "First date" << std::endl;
    std::cout << "Year:";
    std::cin >> y1;
    std::cout << "Month:";
    std::cin >> m1;
    std::cout << "Date:";
    std::cin >> d1;

    std::cout << "Second date" << std::endl;
    unsigned int y2 = 0, m2 = 0, d2 = 0;
    std::cout << "Year:";
    std::cin >> y2;
    std::cout << "Month:";
    std::cin >> m2;
    std::cout << "Date:";
    std::cin >> d2;

    std::cout << "Days between:" << number_of_days(y1, m1, d1, y2, m2, d2) << std::endl;

    using namespace std::chrono_literals;
    std::cout << "Days between:" << number_of_days(2018y / std::chrono::June / 1, 15d / std::chrono::September / 2018) << std::endl;

    return 0;
}
