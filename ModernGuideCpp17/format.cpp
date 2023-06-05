/**
 * @file format.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <format>
#include <iostream>
#include <string>
#include <string_view>

template<typename... Args>
std::string dynamic_print(std::string_view rt_fmt_str, Args &&...args)
{
    return std::vformat(rt_fmt_str, std::make_format_args(args...));
}

/**
 * @brief the Formatting library since C++20,  Prototype from 'fmt' library.
 * 注意编译器版本(GCC13/Clang15)对 C++ 标准里的特性(feature)和库(library)的支持
 * $ g++ format.cpp -std=c++23
 * $ g++ format.cpp -std=c++20
 * $ clang++ format.cpp -std=c++2b
 * $ clang++ format.cpp -std=c++20
 * 
 * $ g++ --version
 * g++.exe (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders) 13.1.0
 * $ clang++ --version
 * clang version 16.0.0 Target: x86_64-pc-windows-msvc Thread model: posix
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // similar to the f-string format in Python.
    std::cout << std::format("Hello {}!\n", "world");

    std::string fmt;
    for (int i{}; i != 3; ++i)
    {
        fmt += "{} "; // constructs the formatting string
        std::cout << fmt << " : ";
        std::cout << dynamic_print(fmt, "alpha", 'Z', 3.14, "unused");
        std::cout << '\n';
    }

    const std::string name{"Wei Li"};
    const char *str = "I love Modern C++";

    const unsigned int age{24};
    auto print_info = std::format("My name is {}, {}, and age is {}\n", name, str, age);
    std::cout << print_info;

    return 0;
}
