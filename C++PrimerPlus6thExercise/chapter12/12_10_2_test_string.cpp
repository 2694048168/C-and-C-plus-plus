/**
 * @file 12_10_2_test_string.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "12_10_2_my_string.hpp"

const int ArSize = 10;
const int MaxLen = 81;

/**
 * @brief 编写C++程序, 测试自定义的 string 类
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // TODO 有 bugging 未解决
    String s1(" and I am a C++ student.");
    String s2 = "Please enter your name: ";
    String s3;

    std::cout << s2;         // overloaded << operator
    std::cin >> s3;          // overloaded >> operator

    s2 = "My name is " + s3; // overloaded =, + operators
    std::cout << s2 << ".\n";
    s2 = s2 + s1;

    s2.string_upper(); // converts string to uppercase
    std::cout << "The string\n" << s2 << "\ncontains " << s2.count_char('A') << " 'A' characters in it.\n";
    s1 = "red"; // String(const char *),

    // then String & operator=(const String&)
    String rgb[3] = {String(s1), String("green"), String("blue")};
    std::cout << "Enter the name of a primary color for mixing light: ";
    String ans;
    bool   success = false;
    while (std::cin >> ans)
    {
        ans.string_lower(); // converts string to lowercase
        for (int i = 0; i < 3; i++)
        {
            if (ans == rgb[i]) // overloaded == operator
            {
                std::cout << "That's right!\n";
                success = true;
                break;
            }
        }
        if (success)
            break;
        else
            std::cout << "Try again!\n";
    }

    std::cout << "Bye\n";

    // ---------------------------------------------------
    // String name;
    // std::cout << "Hi, what's your name?\n>> ";
    // std::cin >> name;

    // std::cout << name << ", please enter up to " << ArSize << " short sayings <empty line to quit>:\n";
    // String sayings[ArSize]; // array of objects
    // char   temp[MaxLen];    // temporary string storage
    // int    i;
    // for (i = 0; i < ArSize; i++)
    // {
    //     std::cout << i + 1 << ": ";
    //     std::cin.get(temp, MaxLen);
    //     while (std::cin && std::cin.get() != '\n') continue;
    //     if (!std::cin || temp[0] == '\0') // empty line?
    //         break;                        // i not incremented
    //     else
    //         sayings[i] = temp; // overloaded assignment
    // }
    // int total = i; // total # of lines read
    // if (total > 0)
    // {
    //     std::cout << "Here are your sayings:\n";
    //     for (i = 0; i < total; i++) std::cout << sayings[i][0] << ": " << sayings[i] << std::endl;
    //     int shortest = 0;
    //     int first    = 0;
    //     for (i = 1; i < total; i++)
    //     {
    //         if (sayings[i].length() < sayings[shortest].length())
    //             shortest = i;
    //         if (sayings[i] < sayings[first])
    //             first = i;
    //     }
    //     std::cout << "Shortest saying:\n" << sayings[shortest] << std::endl;
    //     ;
    //     std::cout << "First alphabetically:\n" << sayings[first] << std::endl;
    //     std::cout << "This program used " << String::HowMany() << " String objects. Bye.\n";
    // }
    // else
    //     std::cout << "No input! Bye.\n";
    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    return 0;
}