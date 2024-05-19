/**
 * @file 03_specialCompilation.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <iostream>

/**
 * @brief 编译中的特别话题
 * 1. 预处理器功能, header file双重包含问题, 以及解决它的方法;
 * 2. 使用编译器标志来优化代码的不同选项;
 * 3. 使用特殊的语言关键字使链接器与 C 语言互操作;
 * 
 * ====预处理器还支持其他指令, 最常见的是宏,
 * 它是一个被赋予名称的代码片段; 每当在C++ 代码中使用这个名字时,预处理器就会用宏的内容取代这个名字.
 * 共有两种不同类型的宏, 即对象类型和函数类型;
 * ?#define ＜NAME＞ ＜CODE＞
 * ?#define <NAME>(<PARAMETERS>) <CODE>
 */
#define MESSAGE "LOL"

// *函数类型的宏与对象类型的宏一样,只是它还可以在标识符之后接受一个参数列表
// #define <NAME>(<PARAMETERS>) <CODE>
#define SAY_LOL_WITH(fn) fn("LOL")

/**
 * @brief Double Inclusion 双重包含
 * 因为只能定义一个符号一次(这个规则被称为 one-definition 规则),
 * 所以必须确保头文件不会试图重复定义符号.
 * 犯这种错误最简单的方法是将同一个头文件包含两次, 这被称为双重包含问题.
 * *避免双重包含问题的常用方法是使用条件编译来创建 include guard 机制,
 * include guard 可以检测头文件是否已经被包含过了; 如果是,它就使用条件编译来清空头文件.
 * 
 */
/*
 // step_function.h
#ifndef STEP_FUNCTION_H 
int step_function(int x);
#define STEP_FUNCTION_H 
#endif
*/

// 大多数现代工具链都支持 #pragma once 的特殊语法，
// 如果支持它的预处理器看到这一行, 它就会表现得像头文件有 include guard 一样
// #pragma once

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "====== Revisiting the Preprocessor =====\n";
    printf(MESSAGE);

    SAY_LOL_WITH(printf);

    /**
     * @brief 条件编译,预处理器还提供条件编译功能,
     * 这是一种提供基本的 if-else 逻辑的工具
     */
    std::cout << "====== Conditional Compilation =====\n";
    if (__cplusplus)
        std::cout << "std::cout\n";
    else
        printf("printf\n");

    /**
     * @brief 编译器优化 Compiler Optimization
     * 现代编译器可以对代码进行复杂的转换, 以提高运行时的性能, 减少二进制文件的大小.
     * 这些转换被称为优化, 它们会让程序员付出一些代价; 优化必然会增加编译时间;
     * 此外, 优化的代码往往比非优化的代码更难调试, 因为优化器通常会消除并重新排列指令;
     * *简而言之, 通常希望在开发时关闭优化功能, 在测试和生产时打开优化功能;
     */

    return 0;
}
