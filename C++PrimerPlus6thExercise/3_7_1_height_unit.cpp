/**
 * @file 3_7_1_height_unit.cpp
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
 * @brief 编写C++程序, 要求用户输入整数身高数值, 单位为英寸(inches); 
 * 然后将英寸转化为英尺(foot)和英寸.
 * 要求使用下划线提示输入位置; 同时使用 const 修饰的常量作为转换因子.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // 1 foot = 12 inches
    // 1 inch = 2.54 cm
    int height = 0;
    std::cout << "Please enter your height:_";
    std::cin >> height;

    const unsigned int scale = 12;

    unsigned int foot = height / scale;
    unsigned int inch = height % 12;

    std::cout << "Your height is " << foot << " foots and " << inch << " inches.\n";

    return 0;
}