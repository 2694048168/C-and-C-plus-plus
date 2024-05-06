/**
 * @file 05_list.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <list>

/**
 * @brief 列表(list)是一个顺序容器, 可在任意位置快速插入元素或删除元素,但无法随机访问元素.
 * STL 实现 std::list 可从＜list＞头文件中获得.
 * 
 * 列表通常被实现为双向链表,即由节点组成的数据结构;
 * 每个节点包含一个元素, 一个前向链接(front-link)和一个后向链接(back-link);
 * 这与将元素存储在连续内存中的向量完全不同, 不能使用 operator[] 或 at 访问列表中的任意元素;
 * *取舍点是因为在列表中插入和删除元素会比较快, 更新的只是元素邻居的 f-link 和 b-link.
 * 
 * 1. 使用 splice 方法将一个列表中的元素拼接到另一个列表中;
 * 2. 使用 unique 方法删除连续的重复元素;
 * 3. 甚至使用 sort 方法对容器的元素进行排序;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::list supports front insertion\n");
    std::list<int> odds{11, 22, 33, 44, 55};

    odds.remove_if([](int x) { return x % 2 == 0; });
    auto odds_iter = odds.begin();
    assert(*odds_iter == 11);
    printf("the first value: %d\n", *odds_iter);

    ++odds_iter;
    assert(*odds_iter == 33);
    ++odds_iter;
    assert(*odds_iter == 55);
    ++odds_iter;
    assert(odds_iter == odds.end());

    return 0;
}
