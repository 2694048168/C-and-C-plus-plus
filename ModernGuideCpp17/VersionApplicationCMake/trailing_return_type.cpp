/**
 * @file trailing_return_type.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 函数返回值类型后置 Trailing Return Type
 * @version 0.1
 * @date 2024-08-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief 在C++的语法糖中，函数的返回值类型一直是函数声明的重要组成部分。
 * C++11 函数返回值类型后置（Trailing Return Type）的语法为开发者提供了新的可能。
 * 这种语法虽然看似是语法上的细微调整，却在现代C++的函数设计和代码可读性方面带来了显著的影响。
 * 
 * 1. 泛型编程和lambda表达式在C++中的应用,传统的返回值声明方式逐渐暴露出一些局限性; 
 * 2. 函数返回值类型后置是C++11引入的新语法,使用 -> 符号将返回值类型放置在参数列表之后;
 *    auto function_name(parameters) -> return_type { // function body}
 *      1. 提高代码可读性
 *      2. 增强模板的灵活性
 *      3. 与现代C++特性的良好兼容性
 * 3. 返回值类型后置的应用场景
 *      1. 函数模板和复杂返回值
 *      2. Lambda表达式与 auto 类型推导
 *      3. 与 decltype 和 std::declval 的结合
 * 
 */

auto add(int a, int b) -> int
{
    return a + b;
}

// 在使用函数模板时，返回值类型可能依赖于模板参数。
// 在这种情况下，返回值类型后置语法显得尤为方便
template<typename T1, typename T2>
auto multiply(T1 a, T2 b) -> decltype(a * b)
{
    return a * b;
}

// 对于一些依赖于表达式结果的返回值类型，
// decltype 和 std::declval 常与返回值类型后置语法结合使用。
template<typename T>
auto getValue(T &t) -> decltype(t.getValue())
{
    return t.getValue();
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "the sum: " << add(12, 42) << '\n';

    std::cout << "the multiple of float: " << multiply(1.2f, 4.2f) << '\n';
    std::cout << "the multiple of int: " << multiply(2, 42) << '\n';

    // Lambda表达式与 auto 类型推导
    auto lambda_func = [](int x) -> int
    {
        return x * 2;
    };
    std::cout << "the double of value: " << lambda_func(24) << '\n';

    return 0;
}
