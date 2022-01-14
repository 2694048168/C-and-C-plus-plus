/**
 * @file literal.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Literal: raw string literal and custom literal
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <string>

std::string operator"" _wow1(const char *wow1, size_t len)
{
    return std::string(wow1) + "woooooooooow, amazing";
}

std::string operator""_wow2(unsigned long long i)
{
    return std::to_string(i) + "woooooooooow, amazing";
}

int main(int argc, char const *argv[])
{
    std::string str = R"(C:\\File\\To\\Path)";
    std::cout << str << std::endl;

    int value = 0b1001010101010;
    std::cout << value << std::endl;

    auto str2 = "abc"_wow1;
    auto num = 1_wow2;
    std::cout << str2 << std::endl;
    std::cout << num << std::endl;
    return 0;
}
