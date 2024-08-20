/**
 * @file vector_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++中如何高效拼接两个vector
 * @version 0.1
 * @date 2024-08-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
  * @brief 在C++编程中,vector是一种常用的数据结构,它代表了一个可以动态改变大小的数组.
  * 在实际开发中, 经常需要将两个vector拼接在一起,形成一个新的vector.
  * 在C++中拼接两个vector有多种方法,
  * 1. 使用insert成员函数
  * 2. push_back和迭代器,
  * 3. 预分配内存
  * 4. 使用C++11的emplace_back
  *
  * 在实际开发中,应根据具体需求和上下文环境选择最合适的方法.
  * 对于性能敏感的应用,建议使用reserve预分配内存,并使用emplace_back减少不必要的元素复制或移动.
  * 
  */

#include <iostream>
#include <vector>

template<typename T>
void printElement(const std::vector<T> &container)
{
    for (const auto &elem : container)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
}

// -----------------------------
int main(int argc, char **argv)
{
    // ============ 一、使用insert成员函数
    // C++ STL中的vector提供了insert成员函数,
    // 可以用来在指定位置前插入另一个容器的全部或部分元素,这是拼接两个vector的一种直观方法.
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {4, 5, 6};
    // 在vec1的末尾插入vec2的所有元素
    vec1.insert(vec1.end(), vec2.begin(), vec2.end());

    // 输出结果
    printElement(vec1);
    /* 性能分析
    使用insert函数进行拼接时,如果vector需要扩展容量, 可能会导致内存重新分配和数据复制,
    从而影响性能. 不过在大多数现代C++实现中, vector的内存分配策略已经相当优化,
    对于不是极端频繁的操作, 这种性能影响通常可以忽略. */

    // ============ 二、使用push_back和迭代器
    // 另一种拼接vector的方法是遍历第二个vector,
    // 并使用push_back函数将其元素逐个添加到第一个vector的末尾。
    // 遍历vec2，将每个元素添加到vec1的末尾
    for (auto it = vec2.begin(); it != vec2.end(); ++it)
    {
        vec1.push_back(*it);
    }
    // 输出结果
    printElement(vec1);
    /* 性能分析
    使用push_back进行拼接时,如果vector的当前容量不足以容纳新元素,
    也会导致内存重新分配. 不过与insert方法相比, push_back通常会有更少的内存复制操作,
    因为它每次只添加一个元素. */

    // ============ 三、使用reserve优化性能
    // 在拼接vector之前,可以先使用reserve函数预分配足够的内存空间
    // 以避免在拼接过程中发生内存重新分配
    // 预分配足够的内存空间
    vec1.reserve(vec1.size() + vec2.size());
    // 使用push_back拼接
    for (auto it = vec2.begin(); it != vec2.end(); ++it)
    {
        vec1.push_back(*it);
    }
    printElement(vec1);
    /* 性能分析
    通过reserve预分配内存,可以确保在拼接过程中不会发生内存重新分配,
    从而提高性能. 这是一种推荐的做法, 尤其是在处理大量数据时. */

    // ============ 四、使用C++11的std::vector::emplace_back
    // C++11引入了emplace_back成员函数,它允许在vector的末尾直接构造元素,
    // 而不是先构造元素再复制到vector中. 这可以减少不必要的元素复制或移动操作,提高性能.
    vec1.reserve(vec1.size() + vec2.size());
    // 使用emplace_back拼接
    for (auto it = vec2.begin(); it != vec2.end(); ++it)
    {
        vec1.emplace_back(*it);
    }
    printElement(vec1);
    /* 性能分析
    emplace_back可以减少不必要的元素复制或移动,因此在拼接包含复杂对象的vector时,
    使用emplace_back可能会带来显著的性能提升. */

    return 0;
}
