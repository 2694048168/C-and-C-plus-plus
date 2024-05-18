/**
 * @file 17_partitioningAlgorithms.cpp
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
 * @brief 分区算法 Partitioning Algorithms
 *  一个分区的序列包含两个连续的、不同的元素组, 
 * 这些组的元素不重合, 第二个组的第一个元素被称为分区点.
 * C++ stdlib 包含了对序列进行分区、确定序列是否已分区, 并找到分区点的算法.
 * 
 */

int main(int argc, const char **argv)
{
    /**
     * @brief is_partitioned 算法确定序列是否已分区
     * 如果目标序列中 pred 评估为 true 的每个元素都出现在其他元素之前,则该算法返回 true
     * ?bool is_partitioned([ep], ipt_begin, ipt_end, pred);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin/ipt_end, 代表目标序列;
     * 3. 确定组成员资格的谓词 pred;
     * *复杂度
     * 线性复杂度, 算法最多调用 pred distance(ipt_begin, ipt_end) 次.
     * 
     */
    std::cout << "\n======== std::is_partitioned algorithm =======\n";
    auto is_odd = [](auto x)
    {
        return x % 2 == 1;
    };

    std::vector<int> numbers1{9, 5, 9, 6, 4, 2};

    bool flag = std::is_partitioned(numbers1.begin(), numbers1.end(), is_odd);
    std::cout << "The sequence is partitioned: " << std::boolalpha << flag << '\n';

    std::vector<int> numbers2{9, 4, 9, 6, 4, 2};
    flag = std::is_partitioned(numbers2.begin(), numbers2.end(), is_odd);
    std::cout << "The sequence is partitioned: " << std::boolalpha << flag << '\n';

    /**
     * @brief partition 算法对序列进行分区,
     * 该算法更改目标序列, 使其根据 pred 进行分区, 它返回分区点,
     * *元素的原始相对顺序不一定被保留.
     * ?ForwardIterator partition([ep], fwd_begin, fwd_end, pred); 
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 ForwardIterator, 即 fwd_begin/fwd_end,代表目标序列;
     * 3. 确定组成员资格的谓词 pred;
     * *复杂度
     * 线性复杂度, 算法最多调用 pred distance(ipt_begin, ipt_end) 次.
     * *其他要求
     * 目标序列的元素必须是可交换的.
     *
     */
    std::cout << "\n======== std::partition algorithm =======\n";
    std::vector<int> numbers3{1, 2, 3, 4, 5};

    const auto partition_point = std::partition(numbers3.begin(), numbers3.end(), is_odd);
    assert(partition_point == numbers3.begin() + 3);
    flag = std::is_partitioned(numbers3.begin(), numbers3.end(), is_odd);
    std::cout << "The sequence is partitioned: " << std::boolalpha << flag << '\n';

    /**
     * @brief partition_copy 算法对序列进行分区,
     * 该算法通过对每个元素进行 pred 评估来划分目标序列.
     * 所有为 true 的元素被复制到 opt_true, 所有为 false 的元素被复制到 opt_false
     * ?ForwardIteratorPair partition_copy([ep], ipt_begin, ipt_end, opt_true, opt_false, pred);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin/ipt_end, 代表目标序列;
     * 3. OutputIterator, 即 opt_true, 用于接收 pred 返回 true 的元素的副本;
     * 4. OutputIterator，即 opt_false, 用于接收 pred 返回 false 的元素的副本;
     * 5. 确定组成员资格的谓词 pred;
     * *复杂度
     * 线性复杂度, 算法恰好调用 pred distance(ipt_begin, ipt_end) 次.
     * *其他要求
     * 1. 目标序列的元素必须是可复制赋值的;
     * 2. 输入序列和输出序列不能重叠;
     * 
     */
    std::cout << "\n======== std::partition_copy algorithm =======\n";
    std::vector<int> numbers4{1, 2, 3, 4, 5}, odds, evens;

    std::partition_copy(numbers4.begin(), numbers4.end(), std::back_inserter(odds), std::back_inserter(evens), is_odd);
    assert(std::all_of(odds.begin(), odds.end(), is_odd));
    assert(std::none_of(evens.begin(), evens.end(), is_odd));

    /**
     * @brief stable_partition 算法对序列进行稳定分区
     * !稳定分区可能比不稳定分区需要更多的计算, 所以用户可以选择适合自己的分区算法
     * 该算法会更改目标序列, 使其根据 pred 进行分区; 元素的原始相对顺序被保留
     * ?BidirectionalIterator partition([ep], bid_begin, bid_end, pred);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 BidirectionalIterator, 即 bid_begin/bid_end, 代表目标序列;
     * 3. 确定组成员资格的谓词 pred;
     * *复杂度
     * 拟线性复杂度, 算法进行 O(N logN) 次交换, N等于 distance(bid_begin, bid_end)
     * 如果内存足够的话, 交换次数为 O(N).
     * *其他要求
     * 目标序列的元素必须是可交换的、可移动构造的，以及可移动赋值的.
     * 
     */
    std::cout << "\n======== std::stable_partition algorithm =======\n";
    std::vector<int> numbers5{1, 2, 3, 4, 5};

    std::stable_partition(numbers5.begin(), numbers5.end(), is_odd);
    for (const auto &elem : numbers5)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;

    return 0;
}
