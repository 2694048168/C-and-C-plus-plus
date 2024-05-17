/**
 * @file 03_findSequence.cpp
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
 * @brief find、find_if 和 find_if_not
 * find、find_if 和 find_if_not 算法在序列中寻找与用户定义的标准相匹配的第一个元素;
 * 这些算法返回指向目标序列的第一个匹配 value(find)的元素的 InputIterator,
 * 结果就是在使用 pred(find_if)调用时产生 true, 在使用 pred(find_if_not)调用时产生 false;
 * 如果算法找不到匹配项, 则返回 ipt_end;
 * ?InputIterator find([ep], ipt_begin, ipt_end, value);
 * ?InputIterator find_if([ep], ipt_begin, ipt_end, pred);
 * ?InputIterator find_if_not([ep], ipt_begin, ipt_end, pred);
 * 
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 InputIterator 对象， 即 ipt_begin / ipt_end， 代表目标序列；
 * 3. 与目标序列的基础类型相当的 const 引用 value(find)或
 *   接受目标序列基础类型的单个参数的谓词(find_if 和 find_if_not);
 *
 * *复杂度
 * 线性复杂度, 该算法最多进行 distance(ipt_begin, ipt_end) 次比较(find)
 *  或 pred 调用(find_if 和 find_if_not).
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]find find_if find_if_not\n";
    std::vector<std::string> words{"fiffer", "feffer", "feff"};

    const auto find_result = std::find(words.cbegin(), words.cend(), "feff");
    assert(*find_result == words.back());
    std::cout << "the find value: " << *find_result << '\n';

    const auto defends_digital_privacy = [](const auto &word)
    {
        return std::string::npos != word.find("eff");
    };

    const auto find_if_result = std::find_if(words.cbegin(), words.cend(), defends_digital_privacy);
    assert(*find_if_result == "feffer");

    const auto find_if_not_result = std::find_if_not(words.cbegin(), words.cend(), defends_digital_privacy);
    assert(*find_if_not_result == words.front());

    /**
     * @brief find_end
     * find_end 算法查找子序列最后一次出现的位置,
     * 如果算法没有找到这样的序列, 则返回 fwd_end1;
     * 如果 find_end 确实找到了一个子序列, 则返回一个指向最后一个匹配子序列的第一个元素的 ForwardIterator
     * ?InputIterator find_end([ep], fwd_begin1, fwd_end1, fwd_begin2, fwd_end2, [pred]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 两对 ForwardIterator, 即fwd_begin1/fwd_end1和fwd_begin2/fwd_end2, 分别代表目标序列1和2;
     * 3. 可选的二元谓词 pred, 用于比较两个元素是否相等;
     * *====复杂度
     * 平方复杂度, 该算法最多进行以下数量的比较或 pred 调用:
     * distance(fwd_begin2, fwd_end2) * (distance(fwd_begin1, fwd_end1)
     *     -  distance(fwd_begin2, fwd_end2) + 1)
     */
    std::cout << "[====]find_end\n";
    std::vector<std::string> words1{"Goat", "girl", "googoo", "goggles"};
    std::vector<std::string> words2{"girl", "googoo"};

    const auto find_end_result1 = std::find_end(words1.cbegin(), words1.cend(), words2.cbegin(), words2.cend());
    assert(*find_end_result1 == words1[1]);

    const auto has_length = [](const auto &word, const auto &len)
    {
        return word.length() == len;
    };

    std::vector<size_t> sizes{4, 6};
    const auto          find_end_result2
        = std::find_end(words1.cbegin(), words1.cend(), sizes.cbegin(), sizes.cend(), has_length);
    assert(*find_end_result2 == words1[1]);

    /**
     * @brief find_first_of
     * find_first_of 算法寻找序列1中第一次出现的等于序列2中的某个元素的元素;
     * 如果提供 pred, 算法会在序列1中找到第一个出现的 i, 其中对于序列2中的某个j,
     * pred (i, j) 为 true;
     * 如果 find_first_of 没有找到这样的序列, 则返回 ipt_end1;
     * 如果 find_first_of 确实找到了一个子序列, 则返回一个指向第一个匹配到的子序列的第一个元素的InputIterator
     * (请注意, 如果 ipt_begin1 也是 ForwardIterator, 则 find_first_of 会返回 ForwardIterator)
     * ?InputIterator find_first_of([ep], ipt_begin1, ipt_end1, fwd_begin2, fwd_end2, [pred]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin1/ipt_end1, 代表目标序列 1;
     * 3. 一对 ForwardIterator, 即 fwd_begin2/fwd_end2, 代表目标序列 2;
     * 4. 可选的二元谓词 pred, 用于比较两个元素是否相等;
     *
     * *复杂度
     * 平方复杂度, 该算法最多进行以下数量的比较或 pred 调用:
     * distance(ipt_begin1, ipt_end1) * distance(fwd_begin2, fwd_end2)
     * 
     */
    std::cout << "[====]find_first_of\n";
    std::vector<std::string> words3{"Hen", "in", "a", "hat"};
    std::vector<std::string> indefinite_articles{"a", "an"};

    const auto find_first_of_result
        = find_first_of(words3.cbegin(), words3.cend(), indefinite_articles.cbegin(), indefinite_articles.cend());

    assert(*find_first_of_result == words3[2]);

    /**
     * @brief adjacent_find
     * adjacent_find 算法寻找序列中的第一对相邻重复元素,
     * 该算法在目标序列中寻找第一次出现相等的两个相邻元素的位置,
     * 如果提供了 pred, 则该算法在序列中寻找 pred （i, i+1）为 true 的第一次出现的元素 i.
     * 如果 adjacent_find 没有找到这样的元素, 则返回 fwd_end;
     * 如果 adjacent_find 找到这样的元素, 将返回一个指向该元素的 ForwardIterator.
     * ?ForwardIterator adjacent_find([ep], fwd_begin, fwd_end, [pred]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 ForwardIterator, 即 fwd_begin/fwd_end, 代表目标序列;
     * 3. 可选的二元谓词 pred, 用于比较两个元素是否相等;
     * *复杂度
     * 线性复杂度, 没有给出执行策略时, 算法最多进行以下次数的比较或 pred 调用:
     * min(distance(fwd_begin, i)+1, distance(fwd_begin, fwd_end)-1) 其中 i 是返回值的索引
     * 
     */
    std::cout << "[====]adjacent_find\n";
    std::vector<std::string> words4{"Icabod", "is", "itchy"};

    const auto first_letters_match = [](const auto &word1, const auto &word2)
    {
        if (word1.empty() || word2.empty())
            return false;
        return word1.front() == word2.front();
    };

    const auto adjacent_find_result = adjacent_find(words4.cbegin(), words4.cend(), first_letters_match);
    assert(*adjacent_find_result == words4[1]);

    return 0;
}
