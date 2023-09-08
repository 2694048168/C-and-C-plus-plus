/**
 * @file 16_10_1_palindrome.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

bool check_palindrome(const std::string &str)
{
    unsigned int low  = 0;
    unsigned int high = str.size() - 1;
    while (low < high)
    {
        char c1 = str[low];
        char c2 = str[high];
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

/**
 * @brief 编写C++程序,简单测试回文字符串
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::string str1{"oto"};

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
    std::string str2{"tot"};

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

    // -------------------------
    std::string str4{"Otto"};

    bool flag4 = check_palindrome(str4);
    if (flag4)
    {
        std::cout << "The string " << str4 << " is palindrome.\n";
    }
    else
    {
        std::cout << "The string " << str4 << " is NOT palindrome.\n";
    }

    // -------------------------
    std::string str5{"otgto"};

    bool flag5 = check_palindrome(str5);
    if (flag5)
    { 
        std::cout << "The string " << str5 << " is palindrome.\n";
    }
    else
    {
        std::cout << "The string " << str5 << " is NOT palindrome.\n";
    }

    return 0;
}