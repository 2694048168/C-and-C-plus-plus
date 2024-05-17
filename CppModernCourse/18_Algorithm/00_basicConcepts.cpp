/**
 * @file 00_basicConcepts.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief 
 * 算法是解决一类问题的过程, 标准库(stdlib)包含许多可以在程序中使用的算法,
 * 因为已有很多优秀的人投入了大量时间确保这些算法的正确性和有效性,
 * 因此通常不需要尝试自己去实现, 样就可以避免"重复造轮子"的事件.
 * ?在开始使用算法之前, 需要对算法复杂度和并行性有一定的了解,
 * ?这两个算法特征是代码执行效率的主要影响因素.
 * 
 * ====算法复杂度 Algorithmic Complexity
 * 算法复杂度描述了计算任务的难度, 量化这种复杂度的一种方法是使用 BachmannLandau 或"大 O"表示法.
 * 大 O 表示法是根据计算量与输入量的关系来描述复杂度的函数,
 * 这种表示法只包括复杂度函数的前项, 前项(leading term)指随着输入规模的增加而增长最快的项.
 * 
 * ----述标准库的算法,这些算法按复杂度可以分为五大类:
 * 1. [Constant time O(1)]固定时间 O(1): 不需要额外计算, 例如确定 std::vector 的大小;
 * 2. [Logarithmic time O(log N)]对数时间 O(log N): 大约计算一次, 例如在 std::set 中查找元素;
 * 3. [Linear time O(N)]线性时间 O(N): 大约 9000 次额外计算, 例如对集合中的所有元素求和;
 * 4. [Quasilinear time O(N log N)]拟线性时间 O(N log N): 大约 37 000 次额外计算, 例如常用的快速排序算法;
 * 4. [Polynomial (or quadratic) time O(N2)]多项式(或平方)时间 O(N2): 大约 99 000 000 次额外计算, 
 *     例如将一个集合中的所有元素与另一个集合中的所有元素进行比较.
 * 
 * ====执行策略 Execution Policies
 * 一些算法可以划分成不同的部分, 以便各独立实体可以同时处理问题的不同部分(这种算法通常称为并行算法).
 * ?许多标准库算法允许使用执行策略指定并行度. [sequentially or in parallel]
 * 执行策略(execution policy)指示算法允许的并行度, 从标准库的角度来看, 算法可以按顺序执行,
 * 也可以并行执行; 顺序算法一次只能有一个实体处理问题, 而并行算法可以让许多实体协同工作以解决问题.
 * *此外,并行算法可以是向量化的, 也可以是非向量化的. [vectorized or non-vectorized]
 * 向量化算法允许实体以未指定的顺序进行工作, 甚至允许单个实体同时处理问题的多个部分.
 * 需要实体之间同步的算法通常是不可向量化的, 因为同一个实体可能多次尝试获取锁, 从而导致死锁.
 * 
 * ----＜execution＞ 头文件中存在三种执行策略:
 * 1. std::execution::seq 指定按顺序(非并行)执行;
 * 2. std::execution::par 指定并行执行;
 * 3. std::execution::par_unseq 指定并行和向量化执行;
 * 
 * 对于那些支持执行策略的算法, 默认是 seq, 这意味着若要使用非顺序执行策略, 必须选择并行度和相关的性能优势.
 * ?请注意, C++ 标准没有指定这些执行策略的确切含义,因为不同的平台处理并行性的方式不同.
 * 当提供非顺序执行策略时, 其实是在声明"此算法可安全并行化".
 * 
 */

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "Welcome the STL Algorithm library\n";

    std::cout << "Please understand the Algorithmic Complexity and Execution Policies\n";

    std::cout << "The best algorithm learning is HackingCpp\n";
    std::cout << "https://hackingcpp.com/cpp/std/algorithms.html\n";

    return 0;
}
