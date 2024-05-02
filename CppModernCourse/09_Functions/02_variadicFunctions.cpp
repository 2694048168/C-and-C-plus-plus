/**
 * @file 02_variadicFunctions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdarg>
#include <cstdint>
#include <cstdio>

/**
 * @brief 可变参数函数 Variadic Functions
 * 1. 可变参数函数接受可变数量的参数, 通常可以通过显式枚举其所有参数来指定函数的参数的确切数量;
 *  使用可变参数函数可以接受任意数量的参数, printf 是一个典型的示例.
 * 
 * TODO: 可变参数函数与Python 中 *args and **kwargs 之间的直接概念关系。
 * 
 * 通过在函数的参数列表中将 ... 作为最后一个参数即可声明可变参数函数;
 * 调用可变参数函数时(invoked), 编译器会将参数与声明的参数进行匹配,
 * 剩余的参数都将打包(pack into)到由 ... 参数表示的可变参数中.
 * !不能直接从可变参数中提取元素, 需使用＜cstdarg＞头文件中的实用函数访问各个参数.
 * 
 * 可变参数函数是 C 的保留项, 通常可变参数函数是不安全的, 并且是安全漏洞的常见来源
 * 可变参数函数至少存在两个主要问题:
 * 1. 可变参数不是类型安全的, 请注意 va_args 的第二个参数是类型;
 * 2. 可变参数中的元素数量必须单独跟踪;
 * 3. 编译器不能解决这两个问题, 可变参数模板提供了一种更安全、更高效的方式来实现可变参数函数.
 * 
 * ===== 可变参数模板 Variadic Templates
 * 通过可变参数模板, 可以创建接受可变参数, 相同类型参数的函数模板, 从而利用模板引擎的强大功能;
 * 要声明可变参数模板, 添加一个特殊的模板参数(称为模板参数包)
 * template ＜typename... Args＞
 * return-type func-name(Args... args) {
 * // Use parameter pack semantics
 * // within function body
 * }
 * 模板参数包是模板参数列表的一部分, 当在函数 template 中使用 Args 时, 它称为函数参数包;
 * 一些可用于参数包的特殊运算符如下:
 * 1. 可以使用 sizeof...(args) 获取参数包的大小;
 * 2. 可以使用特殊语法 other_function(args...) 调用函数(other_function);
 * 3. 此时扩展了参数包 args, 允许对参数包中包含的参数执行进一步的处理.
 * 
 * ==== 用参数包编程(Programming with Parameter Packs)
 * 无法直接在参数包中使用索引, 必须从自身内部调用函数模板(称为编译时递归的过程)
 * *以递归地遍历参数包中的元素.
 * !递归需要一个停止条件, 因此添加了一个不带参数的函数模板特化:
 * template ＜typename T＞
 * void my_func(T x) {
 *  // Use x, but DON'T recurse
 * }
 * 
 * ===== 折叠表达式 Fold Expressions
 * 折叠表达式(fold expression)计算对参数包的所有参数使用二元运算符的结果;
 * 折叠表达式不同于可变参数模板, 但与可变参数模板有关.
 * *(... binary-operator parameter-pack)
 * 
 */

template<typename... T>
constexpr auto sum_fold(T... args)
{
    return (... + args);
}

int sum_(size_t n, ...)
{
    va_list args;
    va_start(args, n);
    int result{};
    while (n--)
    {
        // 使用 va_args 函数遍历可变参数中的每个元素
        auto next_element = va_arg(args, int);
        result += next_element;
    }
    va_end(args);
    return result;
}

// function template specialization without the parameter
template<typename T>
constexpr T sum(T x)
{
    return x;
}

template<typename T, typename... Args>
constexpr T sum(T x, Args... args)
{
    // 它将单个参数 x 从参数包 args 剥离,
    // 然后返回 x 加上 sum 递归调用的结果
    return x + sum(args...);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // 一个是可变参数的数量(6), 后跟六个数字(2、4、6、8、10、12)
    printf("The answer is %d.\n", sum_(6, 2, 4, 6, 8, 10, 12));

    printf("[Variadic Functions]The answer is %d.\n", sum(2, 4, 6, 8, 10, 12));

    printf("[Fold Expressions]The answer is %d.\n", sum_fold(2, 4, 6, 8, 10, 12));

    return 0;
}
