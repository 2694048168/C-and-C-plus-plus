/**
 * @file 06_equalAlgorithm.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief  equal
 * std::equal 算法确定两个序列是否相等,
 * 该算法判断序列 1 的元素是否等于序列 2 的元素;
 * ?bool equal([ep], ipt_begin1, ipt_end1, ipt_begin2, [ipt_end2], [pred]);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 两对 InputIterator, 即 ipt_begin1/ipt_end1 和 ipt_begin2/ipt_end2,
 *    代表目标序列 1 和 2. 如果不提供 ipt_end2, 则意味着序列 1 的长度等于序列 2的长度;
 * 3. 可选的二元谓词 pred, 用于比较两个元素是否相等;
 *
 * *复杂度
 * 线性复杂度, 当没有给出执行策略时, 最坏的情况是算法进行以下数量的比较或 pred调用:
 * min(distance(ipt_begin1, ipt_end1), distance(ipt_begin2, ipt_end2))
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::equal algorithm\n";
    std::vector<std::string> words1{"Lazy", "lion", "licks"};
    std::vector<std::string> words2{"Lazy", "lion", "kicks"};

    const auto equal_result1 = std::equal(words1.cbegin(), words1.cend(), words2.cbegin());
    assert(equal_result1 == false);

    words2[2] = words1[2];

    const auto equal_result2 = std::equal(words1.cbegin(), words1.cend(), words2.cbegin());
    assert(equal_result2);

    /**
     * @brief is_permutation
     * is_permutation 算法确定两个序列是否互为排列,
     * 排列意味着它们包含相同的元素但可能顺序不同.
     * 该算法确定是否存在序列 2 的某种排列, 使得序列 1 的元素等于该排列的元素.
     * ?bool is_permutation([ep], fwd_begin1, fwd_end1, fwd_begin2, [fwd_end2], [pred]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 两对 ForwardIterator, 即 fwd_begin1/fwd_end1 和 fwd_begin2/fwd_end2,
     *   代表目标序列1和2. 如果不提供 fwd_end2, 则意味着序列1 的长度等于序列2 的长度.
     * 3. 可选的二元谓词 pred, 用于比较两个元素是否相等;
     * *复杂度
     * 平方复杂度, 当没有给出执行策略时, 最坏的情况是算法进行以下数量的比较或 pred调用:
     * distance(fwd_begin1, fwd_end1) * distance(fwd_begin2, fwd_end2)
     * 
     * TODO:＜algorithm＞ 头文件还包含 next_permutation 和 prev_permutation
     *  用于操作范围元素,因此可以生成排列.
     */
    std::cout << "[====]std::is_permutation\n";
    std::vector<std::string> words_1{"moonlight", "mighty", "nice"};
    std::vector<std::string> words_2{"nice", "moonlight", "mighty"};

    const auto result = is_permutation(words_1.cbegin(), words_1.cend(), words_2.cbegin());
    assert(result);

    return 0;
}
