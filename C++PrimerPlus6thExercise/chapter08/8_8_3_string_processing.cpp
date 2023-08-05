/**
 * @file 8_8_3_string_processing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cctype>
#include <iostream>
#include <string.>
#include <string>

void str_upper(std::string &str)
{
    for (auto &ch : str)
    {
        ch = toupper(ch);
    }
    std::cout << str << std::endl;
}

/**
 * @brief 编写C++程序, 函数处理字符串, 将用户的输入全部变为大写字符
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::string input;
    std::cout << "Enter a string (q to quit): ";
    std::getline(std::cin, input);
    while (input != "q")
    {
        str_upper(input);

        std::cout << "Next string (q to quit): ";
        std::getline(std::cin, input);
    }

    return 0;
}