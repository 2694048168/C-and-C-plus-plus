/**
 * @file 8_8_1_function_call.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

// void print_string(const std::string *str, int num = 0)
void print_string(const char *str, int num = 0)
{
    // static unsigned long long int num_call;
    static unsigned num_call = 1;

    if (!num)
    {
        std::cout << str;
    }
    else
    {
        for (size_t i = 0; i < num_call; i++)
        {
            std::cout << str;
        }
    }

    ++num_call;
}

/**
 * @brief 编写C++程序, 通过函数打印字符串, 打印次数等于该函数的调用次数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const char* str = "Hello, Wei Li\n";

    print_string(str);
    std::cout << "-------------------------\n";

    print_string(str, 1);
    std::cout << "-------------------------\n";

    print_string(str, 42);
    std::cout << "-------------------------\n";

    return 0;
}