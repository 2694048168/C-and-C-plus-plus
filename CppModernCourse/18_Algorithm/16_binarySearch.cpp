/**
 * @file 16_binarySearch.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <cassert>
#include <ios>
#include <iostream>
#include <vector>

/**
 * @brief 二分搜索 Binary Search
 * !二分搜索算法假定目标序列已经排序, 
 * 与对未假定已排序的序列进行通用搜索相比,这些算法具有理想的复杂度特征.
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief lower_bound 算法在已排序的序列中查找使其分区的元素,
     * 该算法返回一个与元素result 对应的迭代器, 它对序列进行分区, 
     * 因此 result 之前的元素小于 value, 而result 和它之后的所有元素都不小于 value.
     * ?ForwardIterator lower_bound(fwd_begin, fwd_end, value, [comp]);
     * 1. 一对 ForwardIterator, 即 fwd_begin/fwd_end, 代表目标序列;
     * 2. 用来对目标序列分区的 value;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 如果可以提供一个随机迭代器, 就是对数复杂度（O(log N), 
     * N 等于 distance(fwd_begin, fwd_end); 否则, 复杂度为 O(N).
     * *其他要求
     * 目标序列必须根据 operator＜ 或 comp（如果提供的话）排序.
     * 
     */
    std::cout << "========= std::lower_bound algorithm =========\n";
    std::vector<int> numbers{2, 4, 5, 6, 6, 9};

    const auto result = std::lower_bound(numbers.begin(), numbers.end(), 5);
    assert(result == numbers.begin() + 2);
    std::cout << "The value: " << *result << '\n';

    /**
     * @brief upper_bound 算法在有序序列中查找使其分区的元素,
     * 该算法返回一个与元素result 对应的迭代器,这是目标序列中大于 value 的第一个元素.
     * ?ForwardIterator upper_bound(fwd_begin, fwd_end, value, [comp]);
     * 1. 一对 ForwardIterator，即 fwd_begin / fwd_end，代表目标序列;
     * 2. 用来对目标序列分区的 value;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 如果可以提供一个随机迭代器, 就是对数复杂度（O(log N)）,
     *  N 等于 distance(fwd_begin, fwd_end)；否则，复杂度为 O(N).
     * *其他要求
     * 目标序列必须根据 operator＜ 或 comp（如果提供的话）排序.
     * 
     */
    std::cout << "========= std::upper_bound algorithm =========\n";
    std::vector<int> numbers_{2, 4, 5, 6, 6, 9};

    const auto result_ = std::upper_bound(numbers_.begin(), numbers_.end(), 5);
    assert(result_ == numbers_.begin() + 3);
    std::cout << "The value: " << *result_ << '\n';

    /**
     * @brief equal_range 算法在有序序列中查找特定元素的区间,
     * 该算法返回一个迭代器std::pair, 对应于元素等于 value 的半开半闭区间.
     * ?ForwardIteratorPair equal_range(fwd_begin, fwd_end, value, [comp]);
     * 1. 一对 ForwardIterator,即 fwd_begin/fwd_end,代表目标序列;
     * 2. 要寻找的 value;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 如果可以提供一个随机迭代器, 就是对数复杂度（O(log N), 
     * N 等于 distance(fwd_begin, fwd_end);否则,复杂度为 O(N).
     * *其他要求
     * 目标序列必须根据 operator＜ 或 comp（如果提供的话）排序.
     * 
     */
    std::cout << "========= std::equal_range algorithm =========\n";
    std::vector<int> numbers2{2, 4, 5, 6, 6, 9};

    const auto [rbeg, rend] = std::equal_range(numbers2.begin(), numbers2.end(), 6);
    assert(rbeg == numbers2.begin() + 3);
    assert(rend == numbers2.begin() + 5);

    /**
     * @brief binary_search 算法在有序序列中查找特定元素,
     * 如果它包含 value, 则算法返回 true.
     *  具体来说, 如果目标序列存在元素x, 使得它既不满足 x＜value 也不满足 value＜x,
     * 则返回 true. 如果提供了 comp, 则如果目标序列存在元素 x, 
     * 使得comp(x, value) 和 comp(value, x) 均为 false, 则返回 true.
     * ?bool binary_search(fwd_begin, fwd_end, value, [comp]);
     * 1. 一对 ForwardIterator，即 fwd_begin/fwd_end,代表目标序列;
     * 2. 要寻找的 value;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 如果可以提供一个随机迭代器, 就是对数复杂度（O(log N)）,
     * N 等于 distance(fwd_begin, fwd_end); 否则, 复杂度为 O(N).
     * *其他要求
     * 目标序列必须根据 operator＜ 或 comp(如果提供的话)排序.
     * 
     */
    std::cout << "========= std::binary_search algorithm =========\n";
    std::vector<int> numbers3{2, 4, 5, 6, 6, 9};

    bool flag = std::binary_search(numbers.begin(), numbers.end(), 6);
    std::cout << "The value 6 is exist: " << std::boolalpha << flag << '\n';

    flag = std::binary_search(numbers.begin(), numbers.end(), 7);
    std::cout << "The value 7 is exist: " << std::boolalpha << flag << '\n';

    return 0;
}
