/**
 * @file 3_7_2_body_mass_index.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cmath>
#include <iostream>

/**
 * @brief 编写C++程序, 要求用户输入几英尺几英寸的方式的身高，并以磅为单位输入其体重;
 * 计算并报告其BMI数值, 要求使用符号常量表示各种尺度转换因子.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter your height(foot):";
    unsigned int height_foot = 0;
    std::cin >> height_foot;

    std::cout << "Please enter your height(inches):";
    unsigned int height_inches = 0;
    std::cin >> height_inches;

    std::cout << "Please enter your weight(pound):";
    unsigned int weight_pound = 0;
    std::cin >> weight_pound;

    const unsigned foot2inch  = 12;
    const float    inch2metre = 0.0254f;
    const float    pound2kg   = 1 / 2.2f;

    float height = height_foot * foot2inch * inch2metre + height_inches * inch2metre;
    float weight = weight_pound * pound2kg;

    float body_mass_index = std::pow(weight / height, 2);
    std::cout << "Your BMI value is: " << body_mass_index << std::endl;

    return 0;
}