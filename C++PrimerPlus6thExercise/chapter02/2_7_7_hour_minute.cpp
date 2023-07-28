/**
 * @file 2_7_7_hour_minute.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 显示用户输入的时间
 * 
 * @param hours 用户输入的小时数
 * @param minutes 用户输入的分钟数
 */

/**
 * @brief 显示用户输入的时间
 * 
 * @param hours 用户输入的小时数
 * @param minutes 用户输入的分钟数
 * @param remainder 是否对用户输入的数据进行转换, 
 * false表示不进行任何转换; true 表示对分钟数和小时数进行合理转化, 多余小时数直接抹掉!
 */
void show_time(const unsigned int hours, const unsigned int minutes, const bool remainder = false)
{
    if (remainder)
    {
        unsigned add_hours   = minutes / 60;
        unsigned new_minutes = minutes % 60;
        unsigned new_hours   = hours + add_hours;
        new_hours            = new_hours % 24;

        std::cout << "The Time: " << new_hours << ":" << new_minutes << std::endl;
    }
    else
    {
        std::cout << "The Time: " << hours << ":" << minutes << std::endl;
    }
}

/**
 * @brief 编写C++程序, 要求用户输入小时数和分钟数，main 函数调用用户自定义函数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    unsigned hours   = 0;
    unsigned minutes = 0;

    std::cout << "Please enter the number of hours(1-24): ";
    std::cin >> hours;

    std::cout << "Please enter the number of hours(0-60): ";
    std::cin >> minutes;

    // show_time(hours, minutes);

    show_time(hours, minutes, true);

    return 0;
}