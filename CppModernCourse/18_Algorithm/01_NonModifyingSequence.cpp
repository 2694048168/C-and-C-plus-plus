/**
 * @file 01_NonModifyingSequence.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
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
 * @brief Non-Modifying Sequence Operations 非修改序列操作
 * ?非修改序列操作是一种对序列执行计算但不以任何方式修改序列的算法,可以将其视为 const 算法.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief all_of 算法确定序列中的每个元素是否符合用户指定的标准,
     * 如果目标序列为空或序列中所有元素的 pred 为 true, 则算法返回 true;否则, 返回 false.
     * ?bool std::all_of([ep], ipt_begin, ipt_end, pred);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin / ipt_end, 代表目标序列;
     * 3. 一元谓词 pred, 它接受来自目标序列的元素;
     * *====复杂度
     * 线性复杂度,调用 pred 最多 distance(ipt_begin, ipt_end) 次.
     * 
     */
    std::cout << "\n======== std::all_of =======\n";
    std::vector<std::string> words{"Auntie", "Anne's", "alligator"};

    const auto starts_with_a = [](const auto &word)
    {
        if (word.empty())
            return false;
        return word[0] == 'a' || word[0] == 'A';
    };

    bool flag = std::all_of(words.cbegin(), words.cend(), starts_with_a);
    assert(flag == true);

    bool flag_false = std::all_of(words.cbegin(), words.cend(), [](const auto &word) { return word.length() == 6; });
    assert(flag_false == false);

    /**
     * @brief any_of 算法确定序列中是否有元素符合用户指定的标准,
     * 如果目标序列为空或序列中某元素的 pred 为 true, 则算法返回 true; 否则, 返回 false.
     * ?bool any_of([ep], ipt_begin, ipt_end, pred);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin / ipt_end, 代表目标序列;
     * 3. 一元谓词 pred, 它接受来自目标序列的元素;
     * *====复杂度
     * 线性复杂度, 调用 pred 最多 distance(ipt_begin, ipt_end) 次.
     * 
     */
    std::cout << "\n======== std::any_of =======\n";
    std::vector<std::string> words_2{"Barber", "baby", "bubbles"};

    const auto contains_bar = [](const auto &word)
    {
        return word.find("Bar") != std::string::npos;
    };

    assert(std::any_of(words_2.cbegin(), words_2.cend(), contains_bar));
    assert(false == std::any_of(words_2.cbegin(), words_2.cend(), [](const auto &word) { return word.empty(); }));

    /**
     * @brief none_of 算法确定序列中是否没有元素符合用户指定的标准,
     * 如果目标序列为空或序列中没有元素的 pred 为 true, 则算法返回 true; 否则, 返回 false;
     * ?bool none_of([ep], ipt_begin, ipt_end, pred);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin / ipt_end, 代表目标序列;
     * 3. 一元谓词 pred, 它接受来自目标序列的元素;
     * *====复杂度
     * 线性复杂度, 调用 pred 最多 distance(ipt_begin, ipt_end)次;
     * 
     */
    std::cout << "\n======== std::none_of =======\n";
    std::vector<std::string> words_3{"Camel", "on", "the", "ceiling"};

    const auto is_hump_day = [](const auto &word)
    {
        return word == "hump day";
    };

    assert(std::none_of(words.cbegin(), words.cend(), is_hump_day));

    const auto is_definite_article = [](const auto &word)
    {
        return word == "the" || word == "ye";
    };
    bool flag_3 = std::none_of(words_3.cbegin(), words_3.cend(), is_definite_article);
    std::cout << std::boolalpha << flag_3 << '\n';

    return 0;
}
