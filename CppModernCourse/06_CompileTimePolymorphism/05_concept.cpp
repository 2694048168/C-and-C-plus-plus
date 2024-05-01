/**
 * @file 05_concept.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <type_traits>

/**
 * @brief concept 用来约束模板参数, 允许在实例化时而不是首次使用时进行参数检查.
 * 通过在实例化时发现使用问题, 编译器可以提供一个友好的、信息丰富的错误码.
 * 例如试图"用 char* 实例化这个模板, 但这个模板需要一个支持乘法运算的类型."
 * 
 * concept 允许直接在语言中表达对模板参数的要求,
 * GCC6.0 及以后的版本都支持 concept 技术规范,
 * 1. 从根本上改变了实现编译时多态的方式, 熟悉 concept 将带来重大收益;
 * 2. 提供了一个基于 concept 的框架, 用于理解一些临时性的解决方案,
 *    当模板被误用时, 可以把它放到那里获得更容易理解的编译错误;
 * 3. 提供了一个从编译时模板到接口的绝佳桥梁, 接口是运行时多态的主要机制;
 * 4. 使用 GCC6.0 或更高版本, 那么打开 -fconcepts 编译器选项就可以使用 concept;
 * 
 * ===== 定义 concept
 * concept 是一个模板, 一个涉及模板参数的常量表达式, 在编译时进行评估.
 * 把 concept 看成一个大的谓词(predicate): 一个返回值为 false 或 true 的函数.
 * 如果一组模板参数满足给定 concept 的要求, 那么当用这些参数实例化时,
 * 这个 concept 就会被评估为 true, 否则会被评估为 false,
 * !当 concept 被评估为 false 时, 模板实例化就会失败.
 * 可以使用关键字 concept 在熟悉的模板函数定义上声明 concept.
 * 
 * ===== Type Traits 类型特征
 * concept 会验证类型参数, 在 concept 中, 可以操作类型来检查它们的属性;
 * 可以手写这些操作, 也可以使用内置在标准库中的类型支持库, 这个库包含了检查类型属性的实用工具,
 * 这些实用工具被统称为类型特征(type traits) 在＜type_traits＞头文件中, 是std命名空间的一部分.
 * !https://en.cppreference.com/w/cpp/meta#Type_traits
 * 类型特征类 is_integral 和 is_floating_point,
 * 这两个类对于检查类型是整数型还是浮点型很有用
 * 每个类型特征都是一个模板类, 需要一个单一的模板参数, 即要检查的类型;
 * 可以使用模板的静态成员value提取结果,如果类型参数符合要求,则该成员等于true,否则就是false.
 * 
 */

// C++20 support concept
// Declaration of the concept "Hashable", which is satisfied by any type 'T'
// such that for values 'a' of type 'T', the expression std::hash<T>{}(a)
// compiles and its result is convertible to std::size_t
template<typename T>
concept Hashable = requires(T a)
{
    {
        std::hash<T>{}(a)
    } -> std::convertible_to<std::size_t>;
};

// Constrained C++20 function template:
template<Hashable T>
void f(T)
{
}

constexpr const char *as_str(bool x)
{
    return x ? "True" : "False";
}

// -----------------------------------
int main(int argc, const char **argv)
{
    using std::operator""s;
    f("abc"s); // OK, std::string satisfies Hashable

    printf("==================================\n");
    printf("%s\n", as_str(std::is_integral<int>::value));
    printf("%s\n", as_str(std::is_integral<const int>::value));
    printf("%s\n", as_str(std::is_integral<char>::value));
    printf("%s\n", as_str(std::is_integral<uint64_t>::value));
    printf("%s\n", as_str(std::is_integral<int &>::value));
    printf("%s\n", as_str(std::is_integral<int *>::value));
    printf("%s\n", as_str(std::is_integral<float>::value));

    return 0;
}
