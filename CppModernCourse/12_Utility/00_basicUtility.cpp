/**
 * @file 00_basicUtility.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <optional>

/**
 * @brief Utility 工具库
 * C++ stdlib 和 Boost 库提供了大量满足常见编程需求的类型, 类和函数;
 * 这些杂乱无章的工具集合统称为工具库(utility), 除了小,简单和集中的性质之外,工具库在功能上也各不相同.
 * 本章将介绍几种简单的数据结构, 它们可以处理许多常规情况, 在这些情况下, 对象包含其他对象;
 * 本章还将讨论日期和时间, 包括对日历和时钟的编码以及对运行时间的测量, 可用的数字和数学工具.
 * 
 * *数据结构是一种存储对象并允许对这些存储的对象进行某些操作的类型.
 * 1. tribool 是一种类似 bool 的类型, 有三种状态而不是两种状态: true,false,indeterminate
 * 2. optional 是一个类模板, 它包含一个可能存在也可能不存在的值.
 *    optional 的主要用例是为可能失败的函数返回类型, 如果函数成功, 则可以返回一个包含值的 optional,
 *    而不是抛出异常或返回多个值.(stdlib 在＜optional＞头文件中提供了 std::optional)
 * 
 */

struct TheMatrix
{
    TheMatrix(int x)
        : iteration{x}
    {
    }

    const int iteration;
};

enum Pill
{
    Red,
    Blue
};

std::optional<TheMatrix> take(Pill pill)
{
    if (pill == Pill::Blue)
        return TheMatrix{6};
    return std::nullopt;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    printf("std::optional contains types\n");
    if (auto matrix_opt = take(Pill::Blue))
    {
        assert(matrix_opt->iteration == 6);
        auto &matrix = matrix_opt.value();
        assert(matrix.iteration == 6);

        printf("the value: %d, %d\n", matrix_opt->iteration, matrix.iteration);
    }
    else
    {
        printf("The optional evaluated to false.\n");
    }

    printf("std::optional can be empty\n");
    auto matrix_opt_ = take(Pill::Red);
    if (matrix_opt_)
        printf("The matrix is not empty\n");
    printf("the matrix_opt.has_value(): %d", matrix_opt_.has_value());

    return 0;
}
