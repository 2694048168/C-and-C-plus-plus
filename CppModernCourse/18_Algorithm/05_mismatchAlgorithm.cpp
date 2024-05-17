/**
 * @file 05_mismatchAlgorithm.cpp
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
 * @brief mismatch
 * mismatch 算法在两个序列中寻找第一个不匹配的子序列.
 * ?pair<Itr, Itr> mismatch([ep], ipt_begin1, ipt_end1, ipt_begin2, [ipt_end2], [pred]);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 两对 InputIterator, 即 ipt_begin1/ipt_end1 和 ipt_begin2/ipt_end2,
 *   代表目标序列 1 和 2. 如果不提供 ipt_end2, 则意味着序列 1 的长度等于序列 2 的长度;
 * 3. 可选的二元谓词 pred, 用于比较两个元素是否相等;
 * *复杂度
 * 线性复杂度, 当没有给定执行策略时, 最坏的情况是算法进行以下数量的比较或 pred调用:
 * min(distance(ipt_begin1, ipt_end1), distance(ipt_begin2, ipt_end2))
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::mismatch algorithm\n";
    std::vector<std::string> words1{"Kitten", "Kangaroo", "Kick"};
    std::vector<std::string> words2{"Kitten", "bandicoot", "roundhouse"};

    const auto mismatch_result1 = std::mismatch(words1.cbegin(), words1.cend(), words2.cbegin());
    assert(*mismatch_result1.first == "Kangaroo");
    assert(*mismatch_result1.second == "bandicoot");

    const auto second_letter_matches = [](const auto &word1, const auto &word2)
    {
        if (word1.size() < 2)
            return false;
        if (word2.size() < 2)
            return false;
        return word1[1] == word2[1];
    };

    const auto mismatch_result2 = std::mismatch(words1.cbegin(), words1.cend(), words2.cbegin(), second_letter_matches);
    assert(*mismatch_result2.first == "Kick");
    assert(*mismatch_result2.second == "roundhouse");

    return 0;
}
