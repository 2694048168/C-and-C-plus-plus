/**
 * @file 2_7_2_unit_conversion.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 编写 C++ 程序, 要求用户输入距离的数值(单位为 long)，转换为码进行输出
 * 1 long = 220 码(makeweight)
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // std::cout << "please enter an integer for distance(long unit): \n";
    std::cout << "please enter a number for distance(long unit): \n";
    // unsigned int distance = 0;
    float distance = 0.0f;
    std::cin >> distance;

    float unit_conversion = distance * 220;
    std::cout << "after unit conversion, from long unit into makeweight unit,\nthe distance is(makeweight): " << unit_conversion << std::endl;

    return 0;
}