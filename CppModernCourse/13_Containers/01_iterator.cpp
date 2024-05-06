/**
 * @file 01_iterator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <cassert>
#include <cstdio>
#include <iterator>

/**
 * @brief 迭代器 A Crash Course in Iterators
 * 容器和算法之间的接口是迭代器, 迭代器是一种类型.
 * ?它知道容器的内部结构, 并向容器的元素公开简单的指针式操作.
 * 
 * 迭代器有多种风格,但它们都至少支持以下操作：
 * 1）获取当前元素(operator*);
 * 2）移动到下一个元素(operator++);
 * 3）通过赋值使一个迭代器等于另一个迭代器(operator=)
 * 
 * 通过 std::begin 和 std::end 方法从所有 STL 容器(包括 std::array)中提取迭代器;
 * begin 方法返回一个指向第一个元素的迭代器;
 * end 方法返回一个指针, 该指针指向数组最后一个元素后面的元素(为半开半闭区间, half-open range);
 * ?如果容器为空,begin() 将返回与 end() 相同的值;无论容器是否为空, 如果迭代器等于 end(), 则表示已经遍历了容器
 * 
 * TODO: https://en.cppreference.com/w/
 * TODO: https://cplusplus.com/
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::array begin/end form a half-open range\n");
    std::array<int, 0> arr{};
    assert(std::begin(arr) == std::end(arr));
    assert(arr.begin() == arr.end());
    printf("the begin == end for empty array\n");

    printf("\nstd::array can be used as a range expression\n");
    std::array<int, 5> fib{1, 1, 2, 3, 5};

    int sum{};
    for (const auto &element : fib)
    {
        sum += element;
    }
    assert(sum == 12);
    printf("the sum is: %d\n", sum);

    printf("\nstd::array iterators are pointer-like\n");
    std::array<int, 3> easy_as{1, 2, 3};

    auto iter = easy_as.begin();
    assert(*iter == 1);
    ++iter;
    printf("the value of second element: %d\n", *iter);
    ++iter;
    assert(*iter == 3);
    ++iter;
    assert(iter == easy_as.end());

    return 0;
}
