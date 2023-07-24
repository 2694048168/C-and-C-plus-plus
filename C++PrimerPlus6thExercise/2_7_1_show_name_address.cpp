/**
 * @file 2_7_1_show_name_address.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-07-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <string>

/**
 * @brief 编写 C++ 程序, 显示用户姓名和地址
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const char* name = "Wei Li";
    std::string address = "https://github.com/2694048168";

    std::cout << "your name is " << name << "\n";
    std::cout << "your address is " << address << std::endl;

    return 0;
}