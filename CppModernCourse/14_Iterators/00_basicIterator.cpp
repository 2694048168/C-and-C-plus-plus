/**
 * @file 00_basicIterator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <deque>
#include <forward_list>
#include <iterator>
#include <list>
#include <vector>

/**
 * @brief 迭代器是 STL 组件,为容器和操作容器的算法之间提供了接口.
 * 迭代器是类型的接口,它知道如何遍历一个特定的序列, 并暴露针对元素的类似指针的简单操.
 * 每个迭代器至少支持以下操作:
 * 1. 访问当前元素(operator*),进行读或写;
 * 2. 移动到下一个元素(operator++)
 * 3. 复制构造;
 * *迭代器是根据它们支持的额外操作来分类的, 这些类别决定了哪些算法是可用的,以及在泛型代码中可以用迭代器做什么. 
 * 
 * ====迭代器的类别决定了它支持的操作, 这些操作包括读取元素和写入元素,
 * 向前和向后迭代, 多次读取, 以及随机访问元素.
 * *因为接受迭代器的代码通常是通用的, 所以迭代器的类型通常是模板参数, 可以用 concept 来编码.
 * 尽管很可能不必直接与迭代器交互(除非在写一个库), 但仍然需要知道迭代器的类别, 这样就不会将算法应用于不合适的迭代器.
 * 
 * ?1.输出迭代器 Output Iterators 
 * ?2.输入迭代器 Input Iterators
 * ?3.前向迭代器 Forward Iterators
 * ?4.双向迭代器 Bidirectional Iterators
 * ?5.随机访问迭代器 Random-Access Iterators
 * ?6.连续迭代器 Contiguous Iterators
 * ?7.可变迭代器 Mutable Iterators
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    //?=====插入迭代器将写操作转换为容器插入操作
    printf("\nInsert iterators convert writes into container insertions\n");
    std::deque<int> dq;

    auto back_instr = std::back_inserter(dq);
    *back_instr     = 2; // 2
    ++back_instr;
    *back_instr = 4; // 2 4
    ++back_instr;

    auto front_instr = std::front_inserter(dq);
    *front_instr     = 1; // 1 2 4
    ++front_instr;

    auto instr = std::inserter(dq, dq.begin() + 2);
    *instr     = 3; // 1 2 3 4
    instr++;
    assert(dq.at(0) == 1);
    assert(dq.at(1) == 2);
    assert(dq.at(2) == 3);
    assert(dq.at(3) == 4);
    printf("the first and last value: %d, %d\n", dq.at(0), dq[3]);

    //?=====以使用输入迭代器来读取元素、递增和检查相等性
    printf("\nstd::forward_list begin and end provide input iterators\n");
    const std::forward_list<int> easy_as{1, 2, 3};

    auto itr = easy_as.begin();
    assert(*itr == 1);
    itr++;
    assert(*itr == 2);
    itr++;
    assert(*itr == 3);
    itr++;
    assert(itr == easy_as.end());

    //?====前向迭代器也可以多次遍历、默认构造以及复制赋值
    printf("\nstd::forward_list's begin and end provide forward iterators\n");
    auto itr1 = easy_as.begin();
    auto itr2{itr1};

    int double_sum{};
    while (itr1 != easy_as.end())
    {
        double_sum += *(itr1++);
    }
    while (itr2 != easy_as.end())
    {
        double_sum += *(itr2++);
    }
    printf("the sum for twice: %d\n", double_sum);

    //?====双向迭代器是一个也可以向后迭代的前向迭代器
    printf("\nstd::list begin and end provide bidirectional iterators\n");
    const std::list<int> easy_as_list{1, 2, 3};

    auto iter_list = easy_as_list.begin();
    assert(*iter_list == 1);
    ++iter_list;
    assert(*iter_list == 2);
    --iter_list;
    assert(*iter_list == 1);
    assert(iter_list == easy_as_list.cbegin());

    //?====随机访问迭代器是一个支持随机元素访问的双向迭代器
    printf("\nstd::vector begin and end provide random-access iterators\n");
    const std::vector<int> easy_as_vec{1, 2, 3};

    auto iter_vec = easy_as_vec.begin();
    assert(iter_vec[0] == 1);
    ++iter_vec;
    assert(*(easy_as_vec.cbegin() + 2) == 3);
    assert(easy_as_vec.cend() - iter_vec == 2);

    //?====如果迭代器支持读写模式, 那么可以通过解引用迭代器返回的引用赋值,这样的迭代器被称为可变迭代器
    printf("\nMutable random-access iterators support writing\n");
    std::deque<int> easy_as_deque{1, 0, 3};

    auto iter = easy_as_deque.begin();
    iter[1]   = 2;
    ++iter;
    assert(*iter == 2);

    return 0;
}
