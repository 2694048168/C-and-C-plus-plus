/**
 * @file structuredd_binding.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief C++ 17 提供的新特性，结构化绑定; 提供多个返回值(tuple 元组形式)
 * @version 0.1
 * @date 2022-01-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <tuple>
#include <string>

std::tuple<int, double, std::string> multiple_return_function()
{
    return std::make_tuple(1, 2.3, "Wei Li");
}

int main(int argc, char** argv)
{
    auto [x, y, z] = multiple_return_function();
    std::cout << x << ", " << y << ", " << z << std::endl;
    
    return 0;
}
