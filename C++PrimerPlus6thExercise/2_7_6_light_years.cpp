/**
 * @file 2_7_6_light_years.cpp
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
 * @brief 光年距离单位转化为对应的天文单位
 * 
 * @param celsius_degree 摄氏温度
 * @param fahrenheit_degree 华氏温度
 */
void light_astronomical(const double light_years, double &astronomical_units)
{
    astronomical_units = 63240 * light_years;
}

double light_astronomical(const double light_years)
{
    // double astronomical_units = 63240 * light_years;
    // return astronomical_units;

    return 63240 * light_years;
}

/**
 * @brief 编写C++程序, main 函数调用用户自定义函数, 
 * 该函数以光年距离为参数, 并返回对应的天文单位:
 * 1 光年 = 63240 * 天文单位
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "the temperature conversion(from Celsius into Fahrenheit):\n";
    std::cout << "Please enter the number of light years: ";

    double light_years = 0.0;
    std::cin >> light_years;

    double fahrenheit_value = light_astronomical(light_years);

    double astronomical_units = 0.0;
    light_astronomical(light_years, astronomical_units);

    std::cout << light_years << " light years = " << astronomical_units << " astronomical units." << std::endl;

    return 0;
}