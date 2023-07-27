/**
 * @file 4_13_4_person_information.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief 编程C++程序, 以特定形式拼接用户的基本信息
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter your first name? ";
    std::string first_name = "Li";
    std::getline(std::cin, first_name);

    std::cout << "Please enter your last name? ";
    std::string last_name = "Wei";
    std::getline(std::cin, last_name);

    // 如果使用 C-style 字符数组进行处理, 拼接需要考虑存储边界问题!
    std::string person_information = last_name + ", " + first_name;
    std::cout << "Here's the information in a single string: " << person_information << std::endl;

    return 0;
}