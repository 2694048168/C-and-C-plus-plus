/**
 * @file 4_13_2_string_char.cpp
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
 * @brief 修改程序, getline function 的两种用法, 以 newline 为结束符标志.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned array_size = 20;

    char name[array_size];
    char dessert[array_size];

    std::cout << "Please enter your name:\n";
    std::cin.getline(name, array_size);

    std::cout << "Enter your favorite dessert:\n";
    std::cin.getline(dessert, array_size);

    std::cout << "I have some delicious " << dessert;
    std::cout << " for you, " << name << ".\n";

    // -------------------------------------------------
    std::cout << "-----------------------------------\n";
    std::string name_str;
    std::string dessert_str;

    std::cout << "Please enter your name:\n";
    std::getline(std::cin, name_str);

    std::cout << "Enter your favorite dessert:\n";
    std::getline(std::cin, dessert_str);

    std::cout << "I have some delicious " << dessert;
    std::cout << " for you, " << name << ".\n";

    return 0;
}