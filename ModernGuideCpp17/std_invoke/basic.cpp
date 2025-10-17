/**
 * @file basic.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-10-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <functional>
#include <iostream>
#include <string>

/**
 * @brief 调用普通函数、函数指针、Lambda 和函数对象
 * std::invoke的行为等同于标准的函数调用
 * 
 * g++ basic.cpp -std=c++20
 * clang++ basic.cpp -std=c++20
 */

/**
 * @brief 一个简单的普通函数
 * @param name 要打印的名称
 */
void print_name(const std::string &name)
{
    std::cout << "Name: " << name << "\n";
}

// 定义一个函数指针类型
using PrintNamePtr = void (*)(const std::string &);

// 定义一个函数对象
struct Greeter
{
    /**
     * @brief 函数调用操作符
     * @param target 问候的目标
     */
    void operator()(const std::string &target) const
    {
        std::cout << "Hello, " << target << "!" << "\n";
    }
};

// ----------------------------------------
int main(int argc, const char *argv[])
{
    std::string message = "World";

    // 1. 调用普通函数
    std::cout << "1. Calling a free function:\n";
    std::invoke(print_name, "Alice");

    // 2. 调用函数指针
    std::cout << "\n2. Calling a function pointer:\n";
    PrintNamePtr ptr = print_name;
    std::invoke(ptr, "Bob");

    // 3. 调用 Lambda 表达式
    std::cout << "\n3. Calling a lambda:\n";
    auto lambda_printer = [](const std::string &s)
    {
        std::cout << "Lambda says: " << s << std::endl;
    };
    std::invoke(lambda_printer, "Charlie");

    // 4. 调用函数对象
    std::cout << "\n4. Calling a function object:\n";
    Greeter greeter;
    std::invoke(greeter, "Dave");

    return 0;
}
