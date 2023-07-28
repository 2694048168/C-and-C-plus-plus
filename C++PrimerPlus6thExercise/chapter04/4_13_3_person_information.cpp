/**
 * @file 4_13_3_person_information.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstring>
#include <iostream>

/**
 * @brief 编程C++程序, 以特定形式拼接用户的基本信息
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const size_t array_size = 24;

    std::cout << "Please enter your first name? ";
    char first_name[array_size];
    std::cin.getline(first_name, array_size);

    std::cout << "Please enter your last name? ";
    char last_name[array_size];
    std::cin.getline(last_name, array_size);

    // 如果使用 C-style 字符数组进行处理, 拼接需要考虑存储边界问题!
    const char *formatting = ", ";

    char person_information[2 * array_size];
    // strcat(person_information, last_name);
    strcat_s(person_information, last_name);

    // strcat(person_information, formatting);
    strcat_s(person_information, formatting);

    strcat_s(person_information, first_name);

    std::cout << "Here's the information in a single string: " << person_information << std::endl;

    return 0;
}