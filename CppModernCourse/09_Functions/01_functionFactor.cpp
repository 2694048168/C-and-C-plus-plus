/**
 * @file 01_functionFactor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief auto 返回类型
 * 1. Primary: 在函数声明中使用给出明确的返回类型
 * 2. Secondary: 编译器使用 auto 推断正确的返回类型
 * *function signature 最好在可用时提供具体的返回类型.
 * 
 * ---- auto 和函数模板
 * auto 类型推断主要用于函数模板, 其中返回类型(以潜在的复杂方式)取决于模板参数
 * 扩展 auto 返回类型的推断语法, 使用箭头运算符 -＞ 作为后缀, 以提供返回类型,
 * 这样就可以附加一个表达式来计算函数的返回类型(auto 类型推断通常与 decltype 类型表达式配对).
 * ---- decltype 类型表达式会产生另一个表达式的结果类型,
 *  ?decltype(expression) ---> 该表达式解析为表达式的结果类型.
 * TODO:带有模板的泛型编程, 结合 auto 返回类型推断和decltype 可以记录函数模板的返回类型.
 * 
 */
template<typename X, typename Y>
auto add(X x, Y y) -> decltype(x + y)
{
    return x + y;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    auto my_double = add(100., -10);
    printf("decltype(double + int) = double; %f\n", my_double);

    auto my_uint = add(100U, -20);
    printf("decltype(uint + int) = uint; %u\n", my_uint);

    auto my_ulonglong = add(char{100}, 54'999'900ull);
    printf("decltype(char + ulonglong) = ulonglong; %llu\n", my_ulonglong);

    /**
     * @brief Overload Resolution 重载解析
     * Overload resolution is the process that the compiler executes when matching
     * a function invocation with its proper implementation.
     * 
     * ====重载解析是编译器在匹配函数调用与合适的实现时执行的过程.
     * 函数重载可以指定具有相同名称, 不同类型和可能的不同参数的函数,
     * 编译器将函数调用中的参数类型与每个重载声明中的类型进行比较, 从而在这些函数重载中进行选择,
     * 编译器将在可能的选项中选择最佳选项, 如果无法选择最佳选项, 则会报编译错误compiler error.
     * ====大致来说, 匹配过程如下:
     * 1）编译器将寻找完全匹配的类型;
     * 2）编译器将尝试使用整数和浮点类型提升(从int提升到long或从float提升到double)来获得合适的重载;
     * 3）编译器将尝试使用标准类型转换(如将整数类型转换为浮点类型, 将指向子类的指针转换为指向父类的指针)进行匹配;
     * 4）编译器将寻找用户自定义的转换;
     * 5）编译器将查找可变参数函数;
     * 
     */

    return 0;
}
