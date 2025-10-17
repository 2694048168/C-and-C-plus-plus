/**
 * @file meta_programm.cpp
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
#include <type_traits>

/**
 * @brief 模板元编程与高阶函数
 * 在模板元编程中, 经常需要判断一个类型是否可以用一组特定的参数来调用
 * 
 * g++ meta_programm.cpp -std=c++20
 * clang++ meta_programm.cpp -std=c++20
 */

/**
 * @brief 尝试调用一个可调用对象，仅当调用合法时才编译
 * @tparam F 可调用对象的类型
 * @tparam Args 参数类型
 * @param f 可调用对象
 * @param args 参数
 */
template<typename F, typename... Args>
std::enable_if_t<std::is_invocable_v<F, Args...>> try_call_after(F &&f, Args &&...args)
{
    std::cout << "Call is valid. Executing...\n";
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
}

// 一个重载，用于处理无法调用的情况
template<typename F, typename... Args>
std::enable_if_t<!std::is_invocable_v<F, Args...>> try_call_after(F &&, Args &&...)
{
    std::cout << "Call is invalid. Doing nothing.\n";
}

struct MyObject
{
    void foo(int x)
    {
        std::cout << "MyObject::foo(" << x << ")\n";
    }
};

// -------------------------------------
int main(int argc, const char *argv[])
{
    MyObject obj;

    // 合法调用
    try_call_after(&MyObject::foo, obj, 42);

    // 非法调用 (参数数量错误)
    try_call_after(&MyObject::foo, obj);

    // 非法调用 (参数类型错误)
    try_call_after(&MyObject::foo, obj, "hello");

    return 0;
}
