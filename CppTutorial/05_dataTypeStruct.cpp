/**
 * @file 05_dataTypeStruct.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 中基本的数据类型之结构体
 * @version 0.1
 * @date 2024-03-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

struct PersonInfo
{
    std::string  name;
    std::string  englishName;
    unsigned int age;
    bool         gender; // true is Male, false is Female
    std::string  cardID;
    std::string  email;
};

void printInfo(const PersonInfo &info)
{
    std::cout << "========== The Person Info ==========\n";
    std::string gender = info.gender ? "guy" : "girl";
    std::cout << "This " << gender << " name is " << info.name;
    std::cout << " and english name: " << info.englishName;

    std::cout << " and the age is: " << info.age << std::endl;

    std::cout << " you can concat with email:\n\t" << info.email << "\n\n";
}

// ===================================
int main(int argc, const char **argv)
{
    PersonInfo wei;
    wei.name        = "Wei Li";
    wei.englishName = "Ithaca";
    wei.age         = 24;
    wei.gender      = true;
    wei.email       = "weili_yzzcq@163.com";
    wei.cardID      = "8208208820199912122473894";

    printInfo(wei);

    PersonInfo zhou{"Li Zhou", "leta", 21, false, "8208208820200212122473894", "2694048168@qq.com"};
    printInfo(zhou);

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\05_dataTypeStruct.cpp -std=c++23
// g++ .\05_dataTypeStruct.cpp -std=c++23
