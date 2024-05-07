/**
 * @file 11_multisets.cpp
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
#include <set>

/**
 * @brief STL 的＜set＞头文件中可用的 std::multiset 是一个关联容器,
 *  这种容器包含已排序的非唯一键, multiset 支持与普通集合相同的操作, 但它可以存储冗余元素.
 * =====这对两个方法有重要影响:
 * 1. 方法 count 可以返回 0 或 1 以外的值, multiset 的 count 方法可以告诉有多少元素与给定键匹配;
 * 2. 方法 equal_range 可以返回包含多个元素的半开半闭区间, multiset 的 equal_range 方法将返回包含与给定键匹配的所有元素的区间;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::multiset handles non-unique elements\n");
    std::multiset<int> fib{1, 1, 2, 3, 5};

    // "as reflected by size"
    assert(fib.size() == 5);

    // and count returns values greater than 1
    assert(fib.count(1) == 2);
    printf("the count of key=1: %lld\n", fib.count(1));

    // and equal_range returns non-trivial ranges
    auto [begin, end] = fib.equal_range(1);
    assert(*begin == 1);
    ++begin;
    assert(*begin == 1);
    ++begin;
    assert(begin == end);

    return 0;
}
