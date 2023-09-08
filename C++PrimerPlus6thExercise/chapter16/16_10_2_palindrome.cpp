/**
 * @file 16_10_2_palindrome.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

void char_processing(std::string &str);
bool check_palindrome(const std::string &str);

/**
 * @brief 编写C++程序,简单测试回文字符串,考虑大小写和特殊字符等情况
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::string str1{"Otto"};

    bool flag1 = check_palindrome(str1);
    if (flag1)
    {
        std::cout << "The string " << str1 << " is palindrome.\n";
    }
    else
    {
        std::cout << "The string " << str1 << " is NOT palindrome.\n";
    }

    // -------------------------
    std::string str2{"Madam, I'm Adam"}; /* madamimadam */

    bool flag2 = check_palindrome(str2);
    if (flag2)
    {
        std::cout << "The string " << str2 << " is palindrome.\n";
    }
    else
    {
        std::cout << "The string " << str2 << " is NOT palindrome.\n";
    }

    // -------------------------
    std::string str3{"otto"};

    bool flag3 = check_palindrome(str3);
    if (flag3)
    {
        std::cout << "The string " << str3 << " is palindrome.\n";
    }
    else
    {
        std::cout << "The string " << str3 << " is NOT palindrome.\n";
    }

    return 0;
}

void char_processing(const std::string &str, std::string &str_temp)
{
    size_t num_alpha = 0;
    for (size_t i = 0; i < str.size(); ++i)
    {
        if (::isalpha(str[i]))
        {
            str_temp[num_alpha] = str[i];
            ++num_alpha;
        }
    }

    // 字母全部变为大写或者小写，便于比较
    // std::transform(str.cbegin(), str.cend(),
    //                str_temp.begin(), // write to the same location
    //                [](unsigned char c) { return std::toupper(c); });
    std::transform(str_temp.cbegin(), str_temp.cend(), str_temp.begin(), ::tolower);

    // 字符串截断操作
    str_temp = str_temp.substr(0, num_alpha);
}

bool check_palindrome(const std::string &str)
{
    // 输入参数的处理, 利用 pass-by-value，不影响原来的字符串
    std::string str_temp(str.size(), '\0');
    char_processing(str, str_temp);

    // std::cout << "===== " << str_temp << std::endl;

    unsigned int low  = 0;
    unsigned int high = str_temp.size() - 1;
    while (low < high)
    {
        char c1 = str_temp[low];
        char c2 = str_temp[high];
        if (c1 == c2)
        {
            ++low;
            --high;
        }
        else
        {
            return false;
        }
    }

    return true;
}
