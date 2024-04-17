/**
 * @file 07_standardLibrary.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <ostream>
#include <vector>

/**
  * @brief C++标准库(stdlib)是从C语言迁移到C++的一个主要原因, 包含了高性能的泛型代码,
  * stdlib 的三个主要组成部分是容器、迭代器和算法;
  * 1. 容器 是数据结构, 负责保存对象的序列, 容器是正确的、安全的，而且(通常)至少和手写的代码一样有效,
  *   这意味着自己编写容器将花费巨大的精力, 而且不会比stdlib的容器更好, 
  *   容器大体分为两类: 顺序容器和关联容器;
  *   顺序容器在概念上类似于数组, 提供对元素序列的访问权限; 
  *   关联容器包含键值对, 所以容器中的元素可以通过键来查询;
  * 2. 算法 是通用的函数, 用于常见的编程任务, 如计数、搜索、排序和转换,
  *   与容器一样, 算法的质量非常高, 而且适用范围很广, 用户应该很少需要自己实现算法,
  *   而且使用stdlib算法可以极大地提高程序员的工作效率、代码安全性和可读性;
  * 3. 迭代器 可以将容器与算法连接起来, 对于许多stdlib算法的应用, 想操作的数据驻留在容器中,
  *   容器公开迭代器, 以提供平滑、通用的接口, 而算法消费迭代器, 使程序员(包括stdlib的实现者)
  *   不必为每种容器类型实现一个自定义算法
  * 
  */

// -----------------------------------
int main(int argc, const char **argv)
{
    // 现在, 想象一下用C语言编写同等程序所要经历的所有障碍,
    // 就会明白为什么stdlib是一个如此有价值的工具.
    std::vector<int> vec{0, 1, 8, 13, 5, 2, 3};

    vec[0] = 21;
    vec.push_back(1);

    std::sort(vec.begin(), vec.end());

    std::cout << "Printing " << vec.size() << " Fibonacci numbers.\n";
    for (const auto &number : vec)
    {
        std::cout << number << std::endl;
    }

    return 0;
}
