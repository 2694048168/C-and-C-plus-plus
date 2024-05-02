/**
 * @file 03_functionPointers.cpp
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
 * @brief Function Pointers
 * Functional programming is a programming paradigm that
 * emphasizes function evaluation and immutable data. 
 * 
 * 函数式编程是一种强调函数执行和不可变数据的编程范式;
 * 将函数作为参数传递给另一个函数是函数式编程中的主要概念之一;
 * 实现此目的的一种方式是传递函数指针, 函数像对象一样占用内存, 可以通过常用的指针机制来引用该内存地址;
 * 但是不同于对象, 无法修改指向的函数, 函数在概念上类似于 const 对象, 可以获取函数的地址并调用它们.
 * 
 * ---- Declaring a Function Pointer
 * return-type (*pointer-name)(arg-type1, arg-type2, ...);
 * 使用地址运算符 & 来获取函数的地址; 也可以简单地使用函数名称作为指针.
 * 
 * 
 */

float add(float a, int b)
{
    return a + b;
}

// the function signatures match,
// pointer types to these functions will also match
float subtract(float a, int b)
{
    return a - b;
}

// 类型别名和函数指针 Type Aliases and Function Pointers
using operation_func = float (*)(float, int);

// 函数类型, 计算以空字符(null)结尾的字符串中特定字符出现的频率
struct CountIf
{
    CountIf(char x)
        : x{x}
    {
    }

    size_t operator()(const char *str) const
    {
        size_t index{}, result{};
        while (str[index])
        {
            if (str[index] == x)
                result++;
            index++;
        }
        return result;
    }

private:
    const char x;
};

// 将函数对象用作局部应用程序, 自由函数
size_t count_if(char x, const char *str)
{
    size_t index{}, result{};
    while (str[index])
    {
        if (str[index] == x)
            result++;
        index++;
    }
    return result;
}

// -----------------------------------
int main(int argc, const char **argv)

{
    const float first{100};
    const int   second{20};

    float (*operation)(float, int){};
    printf("operation initialized to 0x%p\n", operation);

    operation = &add;
    printf("&add = 0x%p\n", operation);
    printf("%g + %d = %g\n", first, second, operation(first, second));

    operation = &subtract;
    printf("&subtract = 0x%p\n", operation);
    printf("%g - %d = %g\n", first, second, operation(first, second));

    // ========= Type Aliases and Function Pointers =========
    operation_func obj{};
    printf("operation initialized to 0x%p\n", obj);

    obj = &add;
    printf("&add = 0x%p\n", obj);
    printf("%g + %d = %g\n", first, second, obj(first, second));

    obj = &subtract;
    printf("&subtract = 0x%p\n", obj);
    printf("%g - %d = %g\n", first, second, obj(first, second));

    /**
     * @brief 函数调用运算符 The Function-Call Operator
     * 可以通过重载函数调用运算符 operator()() 使用户自定义类型可调用;
     * 这种类型称为函数类型(function type), 函数类型的实例称为函数对象;
     * 函数调用运算符允许参数类型, 返回类型和修饰符(static 除外)的任何组合;
     * 使用户自定义类型可调用的主要原因是希望与期望函数对象使用函数调用运算符的代码进行互操作;
     * 
     * 许多库(例如 stdlib)使用函数调用运算符作为类函数对象的接口,
     * std::async 函数创建异步任务, 该函数接受可以在独立线程上执行的任意函数对象,
     * 它使用函数调用运算符作为接口, std::async 的可能要求你公开某个方法(例如 run 方法),
     * 但它们选择了函数调用运算符, 因为它允许泛型代码使用相同的符号来调用函数或函数对象.
     * 
     */
    /*
    struct type_name
    {
        return_type operator()(arg_type1 arg1, arg_type2 arg2, ...)
        {
            // Body of function-call operator
        }
    }
    */
    CountIf s_counter{'s'};
    auto    sally = s_counter("Sally sells seashells by the seashore.");
    printf("Sally: %zd\n", sally);

    auto sailor = s_counter("Sailor went to sea to see what he could see.");
    printf("Sailor: %zd\n", sailor);
    auto buffalo = CountIf{'f'}(
        "Buffalo buffalo Buffalo buffalo "
        "buffalo buffalo Buffalo buffalo.");
    printf("Buffalo: %zd\n", buffalo);

    printf("========= conceptually similar to function =========\n");
    // employ function objects as partial applications
    auto sally_ = count_if('s', "Sally sells seashells by the seashore.");
    printf("sally_: %zd\n", sally_);

    auto sailor_ = count_if('s', "Sailor went to sea to see what he could see.");
    printf("sailor_: %zd\n", sailor_);

    auto buffalo_ = count_if('f',
                             "Buffalo buffalo Buffalo buffalo "
                             "buffalo buffalo Buffalo buffalo.");
    printf("buffalo_: %zd\n", buffalo_);

    return 0;
}
