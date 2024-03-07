/**
 * @file 04_dataTypeString.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 中基本的数据类型之字符串
 * @version 0.1
 * @date 2024-03-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

// ===================================
int main(int argc, const char **argv)
{
    // 字符串
    std::cout << "============ 字符串 ============\n";
    std::string filepath = R"(D:\Development\PanelTalk)";
    std::string name     = "Wei Li (Ithaca)";

    std::cout << "C++原始字符串: " << filepath << std::endl;
    std::cout << "Your Name via C-style string is: " << name.c_str() << "\n\n";

    std::string strConcat = name + " love C++ programming\n";
    std::cout << "C++字符串拼接操作:\n" << strConcat;

    std::cout << "字符串的长度: " << strConcat.length() << std::endl;
    std::cout << "字符串的大小: " << strConcat.size() << "\n\n";

    bool flag = strConcat.empty() ? true : false;
    std::cout << "字符串是否为空: " << std::boolalpha << flag << std::endl;

    std::string emptyStr;
    bool        flag_ = emptyStr.empty() ? true : false;
    std::cout << "字符串是否为空: " << std::boolalpha << flag_ << std::endl;

    std::cout << "============ 遍历字符串 ============\n";
    for (const auto &elem : strConcat)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\04_dataTypeString.cpp -std=c++23
// g++ .\04_dataTypeString.cpp -std=c++23
