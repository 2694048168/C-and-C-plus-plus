/**
 * @file 9_6_2_character_count.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

// function prototype
void strcount(const std::string &str);

/**
 * @brief 编写C++程序, 完成对字符串中字符的统计功能
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::string input;

    std::cout << "Enter a line: ";
    std::getline(std::cin, input);
    while (input != "")
    {
        strcount(input);

        std::cout << "Enter next line (empty line to quit): ";
        std::getline(std::cin, input);
    }
    std::cout << "\n--------Bye--------\n";

    return 0;
}

void strcount(const std::string &str)
{
    static unsigned total = 0; // static local variable
    unsigned int    count = 0; // automatic local variable

    std::cout << "\"" << str << "\" contains ";
    for (const auto elem : str)
    {
        ++count;
    }

    total += count;

    std::cout << count << " characters\n";
    std::cout << total << " characters total\n";
}