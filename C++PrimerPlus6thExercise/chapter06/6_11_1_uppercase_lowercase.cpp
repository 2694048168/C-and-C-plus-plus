/**
 * @file 6_11_1_uppercase_lowercase.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cctype>
#include <iostream>
#include <string>

/**
 * @brief 编写C++程序, 读入用户的输入, 直到 @ 为止, 进行大小写字母字符转换并显示(忽略数字).
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter some character: ";
    std::string str;
    std::getline(std::cin, str);

    // https://cplusplus.com/reference/cctype/
    for (const auto ch : str)
    {
        if (ch == '@')
        {
            break;
        }
        else
        {
            if (isdigit(ch))
            {
                continue;
            }
            else
            {
                if (isupper(ch))
                {
                    std::cout << (char)tolower(ch);
                }
                else if (islower(ch))
                {
                    std::cout << (char)toupper(ch);
                }
                else
                {
                    std::cout << ch;
                }
            }
        }
    }

    return 0;
}