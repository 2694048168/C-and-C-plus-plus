/**
 * @file 3_7_3_latitude_conversion.cpp
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
 * @brief 编写C++程序, 要求用户以'度，分，秒'方式输入纬度数值，最后以度为单位显示纬度.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter a latitude in degree, minutes, and seconds:\n";
    std::cout << "First, enter the degree:";
    unsigned int latitude_degree = 0;
    std::cin >> latitude_degree;

    std::cout << "Next, enter the minutes of arc:";
    unsigned int latitude_minutes = 0;
    std::cin >> latitude_minutes;

    std::cout << "Finally, enter the seconds of arc:";
    unsigned int latitude_seconds = 0;
    std::cin >> latitude_seconds;

    const float minute2degree = 1.f / 60;
    const float second2minute = 1.f / 60;

    float latitude
        = latitude_degree + latitude_minutes * minute2degree + latitude_seconds * second2minute * minute2degree;

    std::cout << latitude_degree << " degrees, " << latitude_minutes << " minutes, " << latitude_seconds
              << " seconds = " << latitude << " degrees";

    return 0;
}