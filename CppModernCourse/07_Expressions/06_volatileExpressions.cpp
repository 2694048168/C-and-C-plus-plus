/**
 * @file 06_volatileExpressions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief Volatile Expressions
 * volatile 关键字告诉编译器, 通过此表达式进行的每次访问都必须视为可观察的副作用;
 * 这意味着无法对其优化, 也无法通过其他可观察的副作用对其进行重新排序;
 * !此关键字在某些环境(嵌入式编程)中至关重要, 在这些环境中读取和写入内存的某些特殊部分会对底层系统产生影响.
 * volatile 关键字使编译器无法优化此类访问.
 * 
 */

int foo(int &x)
{
    x      = 10;
    x      = 20;
    auto y = x;
    y      = x;
    return y;
}

/**
 * @brief 虽然 x 已被赋值❶, 但在重新赋值❷ 之前从未被使用过, 
 * 因此它被称为无效写入(dead store), 并且它应该直接被优化掉;
 * 类似地, x 在没有任何中间指令❸❹ 的情况下将y 的值设置了两次,
 * 这是冗余读取(redundant load), 也是应优化的对象.
 */
/*
int foo(int &x)
{
    x = 20;
    return x;
}
*/

/**
 * @brief 在一些环境中, 冗余读取和无效写入可能会对系统产生明显的副作用
 * 通过将 volatile 关键字添加到 foo 的参数中, 可以避免优化程序优化掉这些重要的访问.
 */
int foo_volatile(volatile int &x)
{
    x      = 10;
    x      = 20;
    auto y = x;
    y      = x;
    return y;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    int val = 42;
    printf("the value-return: %d\n", foo(val));
    printf("the value-return: %d\n", foo_volatile(val));

    // *@note: https://godbolt.org/z/7esKeK7r7

    return 0;
}
