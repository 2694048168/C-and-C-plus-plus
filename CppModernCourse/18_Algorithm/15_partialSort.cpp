/**
 * @file 15_partialSort.cpp
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
#include <string>
#include <vector>

/**
 * @brief partial_sort 算法将序列分为两组,
 * 如果进行修改, 算法对目标序列中的第一个(rnd_middle - rnd_first)区间的元素进行排序,
 *  使 rnd_begin 到 rnd_middle 的所有元素都小于其他元素.
 * 如果进行复制, 算法将第一个 min(distance(ipt_begin, ipt_end),
 *  distance(rnd_begin,rnd_end)) 排过序的元素放入目标序列, 并返回一个指向输出序列末尾的迭代器.
 * *基本上,partial_sort允许寻找已排序序列的前几个元素, 而无须对整个序列进行排序.
 * ?void partial_sort([ep], rnd_begin, rnd_middle, rnd_end, [comp]);
 * ?RandomAccessIterator partial_sort_copy([ep], ipt_begin, ipt_end, rnd_begin, rnd_end, [comp]);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 如果进行修改, 则使用三个 RandomAccessIterator, 即 rnd_begin/rnd_middle/rnd_end,代表目标序列;
 * 3. 如果进行复制, 则使用 ipt_begin 和 ipt_end 代表目标序列, rnd_begin 和 rnd_end 代表输出序列;
 * 4. 可选的比较运算符 comp;
 * *复杂度
 * 拟线性复杂度 O(N logN), 其中 N 等 于 distance(rnd_begin, rnd_end) * log(distance(rnd_begin，rnd_middle))
 *  或 distance(rnd_begin，rnd_end) * log(min(distance(rnd_begin，rnd_end), distance(ipt_begin，ipt_end)))（后者针对复制版本）
 * *其他要求
 * 目标序列的元素必须是可交换的、可移动构造的，以及可移动赋值的.
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::partial_sort algorithm\n";
    std::string word1{"nectarous"};
    std::partial_sort(word1.begin(), word1.begin() + 4, word1.end());
    std::cout << "the partial_sort: " << word1 << '\n';

    /**
     * @brief is_sorted 算法判断序列是否有序,
     * 如果目标序列根据 operator＜ 或 comp(如果给定)排过序, 则该算法返回 true.
     * is_sorted_until 算法返回一个指向第一个未排序元素的迭代器, 如果目标序列被排序了,则返回 rnd_end.
     * ?bool is_sorted([ep], rnd_begin, rnd_end, [comp]);
     * ?ForwardIterator is_sorted_until([ep], rnd_begin, rnd_end, [comp]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 RandomAccessIterator,即 rnd_begin/rnd_end, 代表目标序列;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 线性复杂度, 该算法比较元素 distance(rnd_end, rnd_begin) 次.
     * 
     */
    std::cout << "[====]std::is_sorted algorithm\n";
    std::string word2{"billowy"};

    bool flag = std::is_sorted(word1.begin(), word1.end());
    std::cout << std::boolalpha << flag << '\n';

    /**
     * @brief nth_element 算法将序列中的特定元素放入其正确的排序位置,
     * 这种部分排序算法通过以下方式修改目标序列: rnd_nth 指向的元素位于所在位置,
     * 就好像正处在已排序过的整个区间的位置一样.
     * 从 rnd_begin 到 rnd_nth-1 的所有元素都将小于 rnd_nth 处的元素.
     * 如果 rnd_nth == rnd_end, 则函数不执行任何操作.
     * ?bool nth_element([ep], rnd_begin, rnd_nth, rnd_end, [comp]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一组 RandomAccessIterator,即 rnd_begin/rnd_nth/rnd_end,代表目标序列;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 线性复杂度, 该算法比较元素 distance(rnd_begin, rnd_end) 次
     * *其他要求
     * 目标序列的元素必须是可交换的、可移动构造的，以及可移动赋值的.
     * 
     */
    std::cout << "[====]std::nth_element algorithm\n";
    std::vector<int> numbers{1, 9, 2, 8, 3, 7, 4, 6, 5};
    std::nth_element(numbers.begin(), numbers.begin() + 5, numbers.end());

    auto less_than_6th_elem = [&elem = numbers[5]](int x)
    {
        return x < elem;
    };
    assert(std::all_of(numbers.begin(), numbers.begin() + 5, less_than_6th_elem));
    std::cout << "The value: " << numbers[5] << '\n';

    return 0;
}
