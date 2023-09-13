/**
 * @file 17_8_1_input_stream.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief 编写C++程序, 计算输入流中第一个 $ 之前的字符数目, 并将 $ 留在输入流中
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::string ch;
    std::cout << "Please enter the character: \n";
    std::getline(std::cin, ch);

    unsigned long num_total = 0;
    for (const auto &elem : ch)
    {
        if (elem == '$')
        {
            break;
        }
        ++num_total;
    }

    std::cout << "The total valid character: " << num_total << "\n";
    std::cout << "The input stream character: " << ch << "\n";

    return 0;
}
