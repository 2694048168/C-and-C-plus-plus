/**
 * @file 2_7_4_age_conversion.cpp
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
 * @brief 编写 C++ 程序, 用户输入年龄, 计算并输出该年龄包含多少个月份
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter your age: " << std::endl;
    unsigned int age = 0;

    // TODO 程序的健壮性, 如何保证用户的输入合法, 不合法的情况下异常如何处理
    std::cin >> age;

    age = age * 12;
    std::cout << "Your age is equal " << age << " months.\n";

    return 0;
}