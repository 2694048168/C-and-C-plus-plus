/**
 * @file 29_availability.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 现代 C++ 语言 可用性增强
 * @version 0.1
 * @date 2026-10-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <iostream>

void foo(char *str)
{
    if (str != nullptr)
        std::cout << str << "\n";
    else
        std::cout << "This is nullptr\n";
}

void foo(int num)
{
    std::cout << num << "\n";
}

constexpr int fibonacci(const int n)
{
    return n == 1 || n == 2 ? 1 : fibonacci(n - 1) + fibonacci(n - 2);
}

template<typename T>
auto process(T value)
{
    if constexpr (std::is_integral_v<T>)
    {
        return value * 2; // 仅对整型编译
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        return value * 2.0; // 仅对浮点型编译
    }
    else
    {
        return std::string(value); // 仅对其他类型编译
    }
}

template<typename T>
void serialize(const T &obj)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        // 算术类型：直接写入二进制
        write_bytes(&obj, sizeof(obj));
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        // 字符串：写入长度 + 内容
        uint32_t len = obj.size();
        write_bytes(&len, sizeof(len));
        write_bytes(obj.data(), len);
    }
    else
    {
        // 其他类型：要求用户提供 serialize 方法
        obj.serialize();
    }
}

// ------------------------------------
int main(int argc, const char *argv[])
{
    foo(0); // 调用 foo(int)
    // foo(NULL);  // 编译失败（二义性）或行为不符合预期
    foo(nullptr); // 调用 foo(char*)

    std::cout << "------------------\n";
    auto ret = fibonacci(12);
    std::cout << "the value of fibonacci(12) = " << ret << "\n";

    std::cout << "------------------\n";
    auto res     = process(12);
    auto res_f   = process(3.14f);
    auto res_str = process("Ithaca");
    std::cout << "the value of process(12) = " << res << "\n";
    std::cout << "the value of process(3.14f) = " << res_f << "\n";
    std::cout << "the value of process(Ithaca) = " << res_str << "\n";

    return 0;
}

// clang++ 29_availability.cpp -std=c++23 -o demo.exe
// g++ 29_availability.cpp -std=c++23 -o demo.exe
