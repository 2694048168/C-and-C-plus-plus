/**
 * @file 10_source_location.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <source_location> // since C++20
#include <string_view>

void log(const std::string_view message, const std::source_location location = std::source_location::current())
{
    std::clog << "file: " << location.file_name() << '(' << location.line() << ':' << location.column() << ") `"
              << location.function_name() << "`: " << message << '\n';
}

template<typename T>
void fun(T x)
{
    log(x); // line 25
}

// -----------------------------------
int main(int argc, const char **argv)
{
    log("Hello world!"); // line 31
    fun("Hello C++20!");

    // main-stream compiler and c++lib supported the feature:
    // g++ 10_source_location.cpp -std=c++20
    // clang++ 10_source_location.cpp -std=c++20
    // cl 10_source_location.cpp /std:c++20 /EHsc

    return 0;
}
