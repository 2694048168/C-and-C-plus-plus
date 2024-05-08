/**
 * @file 01_auxiliaryIteratorFunctions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iterator>
#include <vector>

/**
 * @brief Auxiliary Iterator Functions 迭代器辅助函数
 * 如果要使用迭代器编写泛型代码, 则应该使用＜iterator＞头文件的迭代器辅助函数来操作迭代器,
 * 而不是直接使用迭代器. 这些迭代器函数执行遍历、交换和迭代器之间距离的计算等常见任务,
 * 使用辅助函数而不是直接操作迭代器的主要优势是, 辅助函数可以检查迭代器的类型特征,
 * 并确定执行所需操作最有效的方法.
 * 此外, 迭代器辅助函数使泛型代码更加通用, 因为它适用于广泛的迭代器.
 * 
 * ?1====std::advance
 * std::advance 辅助函数允许按所需的数量递增或递减, 
 * 这个函数模板接受一个迭代器引用和一个对应于想要移动迭代器的距离的整数值
 * ?2====std::next 和 std::prev
 * std::next 和 std::prev 这两个迭代器辅助函数是计算从给定迭代器出发的偏移量的函数模板,
 * 它们返回一个指向所需元素的新迭代器, 而不需要修改原来的迭代器.
 * ?3=====std::distance
 * std::distance 迭代器辅助函数可以计算两个输入迭代器 itr1 和 itr2 之间的距离
 * ?====std::iter_swap
 * std::iter_swap 迭代器辅助函数允许交换两个前向迭代器 itr1 和 itr2 所指向的值
 * *迭代器不需要有相同的类型, 只要它们所指向的类型可以相互赋值即可
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nadvance modifies input iterators\n");
    std::vector<unsigned char> mission{0x9e, 0xc4, 0xc1, 0x29, 0x49, 0xa4, 0xf3, 0x14,
                                       0x74, 0xf2, 0x99, 0x05, 0x8c, 0xe2, 0xb2, 0x2a};

    auto iter = mission.begin();
    std::advance(iter, 4);
    assert(*iter == 0x49);
    std::advance(iter, 4);
    assert(*iter == 0x74);
    printf("the value: %c\n", *iter);
    std::advance(iter, -8);
    assert(*iter == 0x9e);

    printf("\nnext returns iterators at given offsets\n");
    auto itr1 = mission.begin();
    std::advance(itr1, 4);
    assert(*itr1 == 0x49);

    auto itr2 = std::next(itr1);
    assert(*itr2 == 0xa4);
    auto itr3 = std::next(itr1, 4);
    assert(*itr3 == 0x74);

    assert(*itr1 == 0x49);

    printf("\ndistance returns the number of elements between iterators\n");
    auto eighth = std::next(mission.begin(), 8);
    auto fifth  = std::prev(eighth, 3);
    assert((std::distance(fifth, eighth) == 3));
    printf("the distance of tow iterator: %lld\n", std::distance(fifth, eighth));

    printf("\niter_swap swaps pointed-to elements\n");
    std::vector<long> easy_as{3, 2, 1};
    std::iter_swap(easy_as.begin(), std::next(easy_as.begin(), 2));

    assert(easy_as[0] = 1);
    assert(easy_as[1] = 2);
    assert(easy_as[2] = 3);

    return 0;
}
