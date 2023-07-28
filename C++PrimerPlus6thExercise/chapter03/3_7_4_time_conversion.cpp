/**
 * @file 3_7_4_time_conversion.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 编写C++程序, 要求用户以整数方式输入时间秒数, 然后计算为天, 时分秒显示.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter the number of seconds: ";
    unsigned long long total_seconds = 0;
    std::cin >> total_seconds;

    const unsigned int scale     = 60;
    const unsigned int scale_day = 24;

    unsigned int       seconds       = total_seconds % scale;
    unsigned long long total_minutes = total_seconds / scale;

    unsigned int       minutes     = total_minutes % scale;
    unsigned long long total_hours = total_minutes / scale;

    unsigned int hours = total_hours % scale;
    unsigned int days  = total_hours / scale_day;

    std::cout << total_seconds << " seconds = " << days << " days, " << hours << " hours, " << minutes << " minutes, "
              << seconds << " seconds" << std::endl;

    return 0;
}