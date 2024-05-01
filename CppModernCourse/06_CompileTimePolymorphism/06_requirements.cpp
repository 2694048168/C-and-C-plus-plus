/**
 * @file 06_requirements.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <stdexcept>
#include <type_traits>

/**
 * @brief 约束要求(requirements) 是对模板参数的临时约束,
 * 每个 Concept 都可以对其模板参数指定任意数量的约束要求,
 * 约束要求被写成 requires 表达式, 用 requires 关键字后跟函数参数和函数体表示.
 * 约束要求的语法序列构成了 requires 表达式的主体, 每一个符合语法的约束要求都对模板参数进行了约束.
 * 综合起来, requires 表达式有如下形式:
 * requires (arg-1, arg-2, ...) {
 *  { expression1 } -＞ return-type1;
 *  { expression2 } -＞ return-type2;
 *  // --snip--
 *  }
 * 
 * requires 表达式接受放在 requires 关键字之后的参数,这些参数的类型来自模板参数.
 * 随后是约束要求, 每个约束要求用 { } -＞ 表示, 每个大括号内可以放入任意表达式,
 * 这个表达式可以拥有符合参数表达式的任意数量的参数.
 * 如果某实例化导致表达式不能被编译, 则该约束要求不满足; 假设表达式求值没有问题,
 *  则下一步检查该表达式的返回类型是否与 -＞ 后面给出的类型相匹配, 
 *  如果表达式结果的类型不能隐式地转换为返回类型, 则该约束要求视为不满足.
 * *如果任何一个语法约束要求不满足, 则 requires 表达式被评估为false,
 * *如果所有的语法约束要求都通过检查了, 则 requires 表达式被评估为true.
 * 
 * https://en.cppreference.com/w/cpp/language/requires
 * 
 */
// T, U are types
/*
requires(T t, U u)
{
    {
        t == u
    } -> bool; // syntactic requirement 1
    {
        u == t
    } -> bool; // syntactic requirement 2
    {
        t != u
    } -> bool; // syntactic requirement 3
    {
        u != t
    } -> bool; // syntactic requirement 4
}
*/

/**
 * @brief 从 requires 表达式构建 concept
 * 因为 requires 表达式是在编译时求值的, 所以 concept 可以包含任意数量的表达式.
 * 可以试着构造一个 concept, 防止误用 mean.
 * 
 * 这段代码所隐含的三个约束要求
 * 1. T 必须是默认可构造的;
 * 2. T 支持 operator+=;
 * 3. 将 T 除以 size_t, 得到的仍然是 T;
 * 
 */

/*
template<typename T>
T mean(T *values, size_t length)
{
    T result{};
    for (size_t i{}; i < length; ++i)
    {
        result += values[i];
    }
    return result / length;
}
*/

// concept 只是一些谓词: 正在构建一个布尔表达式
template<typename Type>
concept Averageable = std::is_default_constructible<Type>::value && requires(Type a, Type b)
{
    a += b;
};

// 使用 concept
// 声明 concept 比使用 concept 要麻烦得多,
// 要使用 concept，只需用 concept 的名称来代替 typename 关键字
template<Averageable T>
T mean(const T *values, size_t length)
{
    T result{};
    for (size_t i{}; i < length; ++i)
    {
        result += values[i];
    }
    return result / length;
}

//  临时 requires 表达式
// concept 是相当重量级的强类型安全机制,
// 有时只想直接在模板前缀中增加一些约束要求
// 这可以通过直接在模板定义中嵌入 requires 表达式来实现
template<typename T>
requires std::is_copy_constructible<T>::value T get_copy(T *pointer)
{
    if (!pointer)
        throw std::runtime_error{"Null-pointer dereference"};
    return *pointer;
}

// ----------------------------------
int main(int argc, const char **argv)
{
    double     nums_d[]{1.0f, 2.0f, 3.0f, 4.0f};
    const auto result1 = mean(nums_d, 4);
    printf("double: %f\n", result1);

    float      nums_f[]{1.0, 2.0, 3.0, 4.0};
    const auto result2 = mean(nums_f, 4);
    printf("float: %f\n", result2);

    size_t     nums_c[]{1, 2, 3, 4};
    const auto result3 = mean(nums_c, 4);
    printf("size_t: %zd\n", result3);

    return 0;
}
