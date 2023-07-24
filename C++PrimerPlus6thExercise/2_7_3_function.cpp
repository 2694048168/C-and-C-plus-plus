/**
 * @file 2_7_3_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string_view>

/**
 * @brief 以指定的次数重复输出指定的内容
 * 
 * @param msg 需要输出的指定内容
 * @param count 需要重复的次数
 */
void print_message(const char *msg, const unsigned int count)
{
    for (size_t i = 0; i < count; ++i)
    {
        std::cout << msg << "\n";
    }
}

void print_message(const char *msg)
{
    std::cout << msg << "\n";
}

/**
 * @brief 以指定的次数重复输出指定的内容
 * 
 * @param msg 需要输出的指定内容
 * @param count 需要重复的次数
 */
void display_information(std::string_view msg, const unsigned int count)
{
    for (size_t i = 0; i < count; ++i)
    {
        std::cout << msg << "\n";
    }
}

void display_information(std::string_view msg)
{
    std::cout << msg << "\n";
}

/**
 * @brief 编写 C++ 程序, 通过自定义函数形式输出指定内容
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const char        *message     = "Three blind mice";
    std::string_view   information = "See how they run";
    const unsigned int count       = 2;

    print_message(message, count);
    display_information(information, count);

    std::cout << "---------------------------" << std::endl;
    print_message(message);
    print_message(message);
    display_information(information);
    display_information(information);

    return 0;
}