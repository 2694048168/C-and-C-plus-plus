/**
 * @file 04_number_string.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

// ------------------------------
int main(int argc, char *argv[])
{
    // ========= 数值转换为字符串 =========
    // header <string> 'to_string()' function
    std::string pi   = "pi is " + std::to_string(3.1415926);
    std::string love = "love is " + std::to_string(5.20 + 13.14);
    std::cout << pi << "\n" << love << std::endl;

    // ========= 字符串转换为数值 =========
    std::cout << "------------------------\n";
    // int stoi(str, ...);
    // float stof(str, ...);
    // double stod(str, ...);
    std::string str1 = "45";
    std::string str2 = "3.14159";
    std::string str3 = "9527 with words";
    std::string str4 = "words and 2"; 

    int   num1 = std::stoi(str1);
    float num2 = std::stof(str2);
    int   num3 = std::stoi(str3);
    // int   num4 = std::stoi(str4); /* running time error */
    // terminate called after throwing an instance of 'std::invalid_argument'

    std::cout << "std::stoi(\"" << str1 << "\") is " << num1 << std::endl;
    std::cout << "std::stof(\"" << str2 << "\") is " << num2 << std::endl;
    std::cout << "std::stoi(\"" << str3 << "\") is " << num3 << std::endl;
    // std::cout << "std::stoi(\"" << str4 << "\") is " << num4 << std::endl;

    return 0;
}