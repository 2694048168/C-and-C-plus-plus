/**
 * @file 12_fold_expression.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 折叠表达式 Fold Expressions
 * @version 0.1
 * @date 2024-10-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief Fold expression C++17 中引入的一项新特性,专门用于简化可变参数模板的操作.
 * 在没有折叠表达式之前, 处理可变参数的递归模板函数可能非常复杂,
 * 而折叠表达式通过将参数组合起来大大简化了这一过程.
 * 
 * *1.为什么需要折叠表达式?
 * 在 C++11 和 C++14 中, 处理可变参数模板通常需要递归展开模板函数;
 * 这种递归方法代码冗长、难以维护，尤其是在处理大量参数时.
 * 
 * *2.折叠表达式的形式
 * 折叠表达式通过将参数用指定的运算符组合起来, 根据运算符的位置, 可以分为以下几种折叠形式:
 * ?----右折叠: (args +...), 从左到右展开;
 * ?----左折叠: (... + args), 从右到左展开;
 * ?----包裹折叠: (init + ... + args), 通过初始化值开始进行折叠;
 * ?----混合折叠: (args + ... + init), 从左侧起始进行折叠, 并以初始值结束;
 * 
 * 总结：
1. 包裹折叠 (init + … + args)：
• 从初始值 init 开始，依次折叠参数，并从左向右进行累加，最后返回结果。
• 例如：(10 + 1 + 2 + 3) 会变成 (((10 + 1) + 2) + 3)。
2. 混合折叠 (args + … + init)：
• 从参数 args... 开始依次累加，最后将结果与初始值 init 相加。
• 例如：(1 + 2 + 3 + 10) 会变成 ((1 + 2) + 3) + 10。
使用场景：
1. 包裹折叠：
• 当你希望从一个初始值开始，并且依次累加所有参数时，包裹折叠是合适的选择。例如，可以用于从某个起始值开始进行
累加、乘法等。
2. 混合折叠：
• 当你希望首先处理参数，然后在最后附加一个初始值时，混合折叠是更好的选择。例如，可以在计算完所有参数的和后，
将结果再加上一个固定的附加值。
 * 
 * *3.使用场景示例
 * ?1.示例代码-右折看的求和操作;
 * 
 * *4.折叠表达式的意义
 * ----i.简洁性:折叠表达式通过简化可变参数模板的展开, 极大地减少了代码的复杂度;
 * 例如通过 (args +...)实现了对任意数量参数的求和; 如果没有折叠表达式, 必须编写递归模板来处理每一个参数;
 * ----ii.可读性:折叠表达式的语法非常直观, 能够清晰地展示参数之间的运算逻辑, 使代码更加易读;
 * ----iii.减少递归模板:传统的处理可变参数的方式通常依赖递归模板, 这种方法可能导致编译器生成庞大的代码,
 * 而且维护起来不方便.折叠表达式通过内建的展开机制, 避免了递归的需求;
 *
 * *5.折叠表达式解决的问题
 * ----减少模板展开的复杂度: 通过折叠表达式, 可以轻松地对任意数量的参数进行运算, 不再需要使用递归模板;
 * ----简化可变参数的处理: 在进行类似求和、逻辑运算或组合多个参数的场景中, 折叠表达式使得处理流程更加清晰,
 * 并且可以直接使用常见的运算符(如+，*，&&，等);
 * ----提高可维护性: 由于折叠表达式使代码更加简洁, 它在处理复杂模板时有助于提高代码的可维护性, 减少出错的可能性;
 * 
 */

//C++11 or C++14中 处理可变参数模版通常是需要递归展开模版函数
/*
递归展开
递归展开：在 main() 函数中调用 sum(1, 2, 3, 4) 时，递归模板会依次展开：
• sum(1, 2, 3, 4) 展开为 1 + sum(2, 3, 4)
• sum(2, 3, 4) 展开为 2 + sum(3, 4)
• sum(3, 4) 展开为 3 + sum(4)
• sum(4) 展开为 4 + sum()
• sum() 返回 0于是得到 1 + 2 + 3 + 4 + 0 = 10
*/

int sum()
{
    return 0;
}

template<typename T, typename... Args>
T sum(T first, Args... args)
{
    return first + sum(args...);
}

//C++17 使用折叠表达式处理可变参数模版
template<typename... Args>
auto sum_with_init(int init, Args... args)
{
    return (init + ... + args); // 包裹折叠, 从 init 开始累加所有参数
}

template<typename... Args>
auto sum_with_final(int init, Args... args)
{
    return (args + ... + init); // 混合折叠, 参数与初始值相加, 初始值在最后
}

//C++17中可以直接使用折叠表达式
/*
右折叠表达式：
• (args + ...) 是一个右折叠表达式。
右折叠意味着运算从左到右依次展开，最右边的运算符最先与最后一个参数进行结合
当 sum 函数接收一系列参数（比如 1, 2, 3, 4），右折叠表达式会依次展开为： (1 + (2 + (3 + 4)))
这个展开过程可以理解为：
• 先计算 (3 + 4)，结果为 7
• 然后计算 2 + 7，结果为 9
• 最后计算 1 + 9，结果为 10
这就是右折叠表达式的特点，它从右边开始逐渐展开并结合。
*/
template<typename... Args>
auto sum17(Args... args)
{
    return (args + ...);
}

// =====================================
int main(int argc, const char **argv)
{
    std::cout << "Sum of 1, 2, 3, 4: " << sum(1, 2, 3, 4) << '\n';

    // sum_with_init(10, 1, 2, 3) 将被展开 (((10 + 1) + 2) + 3)
    std::cout << "sum_with_init(10, 2, 3, 4): " << sum_with_init(10, 2, 3, 4) << '\n';

    // sum_with_final(10, 1, 2, 3) 将被展开 (((1 + 2) + 3) + 10)
    std::cout << "sum_with_final(10, 2, 3, 4): " << sum_with_final(10, 2, 3, 4) << '\n';

    std::cout << "Sum17 of 1, 2, 3, 4: " << sum17(1, 2, 3, 4) << '\n';

    return 0;
}