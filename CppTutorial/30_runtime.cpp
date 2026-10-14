/**
 * @file 30_runtime.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 现代 C++ 运行期 优化
 * @version 0.1
 * @date 2026-10-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <functional>
#include <iostream>
#include <memory>
#include <string>

int foo(int para)
{
    return para;
}

int foo_bind(int a, int b, int c)
{
    return a + b + c;
}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// ------------------------------------
int main(int argc, const char *argv[])
{
    auto println = [](const std::string &str)
    {
        std::cout << str << "\n";
    };

    println("hello world");

    println("--------------------");
    auto add = [](auto x, auto y)
    {
        return x + y;
    };
    std::cout << "the value = " << add(1, 2) << std::endl;
    std::cout << "the value = " << add(1.1, 2.2) << std::endl;

    println("--------------------");
    std::function<int(int)> func      = foo;
    int                     important = 10;
    std::function<int(int)> func2     = [&](int value) -> int
    {
        return 1 + value + important;
    };
    std::cout << "the value =" << func(42) << std::endl;  // 42
    std::cout << "the value =" << func2(10) << std::endl; // 21

    auto bindFoo = std::bind(foo_bind, std::placeholders::_1, 1, 2);
    std::cout << "the value =" << bindFoo(1) << std::endl; // 输出 4

    println("--------------------");
    std::string              str = "Hello world.";
    std::vector<std::string> v;

    // 使用 push_back(const T&)，产生拷贝行为
    v.push_back(str);
    std::cout << "str: " << str << std::endl; // str: Hello world.

    // 使用 push_back(const T&&)，不会出现拷贝行为
    v.push_back(std::move(str));
    std::cout << "str: " << str << std::endl; // str: （空）

    return 0;
}

// clang++ 30_runtime.cpp -std=c++23 -o demo.exe
// g++ 30_runtime.cpp -std=c++23 -o demo.exe
