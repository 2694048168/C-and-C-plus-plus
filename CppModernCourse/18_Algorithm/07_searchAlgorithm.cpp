/**
 * @file 07_searchAlgorithm.cpp
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
 * @brief search
 * search 算法可以定位子序列, 该算法在序列 1 中定位序列 2;
 * 换句话说, 它返回序列1 中的第一个迭代器 i, 这样对于每个非负整数 n,
 *  有 *(i + n) 等 于 *(ipt_begin2 + n).
 * !这与 find 不同, 因为它定位子序列而不是单个元素.
 * ?ForwardIterator search([ep], fwd_begin1, fwd_end1, fwd_begin2, fwd_end2, [pred]);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 两对 ForwardIterator, 即 fwd_begin1/fwd_end1 和 fwd_begin2/fwd_end2, 分别代表目标序列 1 和 2;
 * 3. 可选的二元谓词 pred, 用于比较两个元素是否相等;
 * *复杂度
 * 平方复杂度, 当没有给出执行策略时, 最坏的情况是算法进行以下数量的比较或 pred调用:
 * distance(fwd_begin1, fwd_end1) * distance(fwd_begin2, fwd_end2)
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::search algorithm\n";
    std::vector<std::string> words1{"Nine", "new", "neckties", "and", "a", "nightshirt"};
    std::vector<std::string> words2{"and", "a", "nightshirt"};

    const auto search_result_1 = std::search(words1.cbegin(), words1.cend(), words2.cbegin(), words2.cend());
    assert(*search_result_1 == "and");
    std::cout << "The search value: " << *search_result_1 << '\n';

    std::vector<std::string> words3{"and", "a", "nightpant"};
    const auto search_result_2 = std::search(words1.cbegin(), words1.cend(), words3.cbegin(), words3.cend());
    assert(search_result_2 == words1.cend());

    /**
     * @brief search_n
     * search_n 算法定位包含相同连续值的子序列,
     * 该算法在序列中搜索 count 个连续值并返回一个指向第一个值的迭代器,
     * 如果没有找到这样的子序列, 则返回 fwd_end;
     * !这与 adjacent_find 不同, 因为它定位子序列而不是单个元素.
     * ?ForwardIterator search([ep], fwd_begin, fwd_end, count, value, [pred]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 ForwardIterator, 即 fwd_begin/fwd_end, 代表目标序列;
     * 3. 表示要查找的连续匹配数的整数 count;
     * 4. 代表要查找的元素的值;
     * 5. 可选的二元谓词 pred, 用于比较两个元素是否相等;
     * *====复杂度
     * 线性复杂度, 当没有给出执行策时, 最坏的情况是算法进行 distance(fwd_begin, fwd_end) 次比较或 pred 调用
     * 
     */
    std::cout << "[====]search_n algorithm\n";
    std::vector<std::string> words_{"an", "orange", "owl", "owl", "owl", "today"};

    const auto result = std::search_n(words_.cbegin(), words_.cend(), 3, "owl");

    assert(result == words_.cbegin() + 2);

    return 0;
}
