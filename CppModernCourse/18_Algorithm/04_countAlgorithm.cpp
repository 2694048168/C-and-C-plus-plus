/**
 * @file 04_countAlgorithm.cpp
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
 * @brief std::count
 * count 算法对序列中匹配某些用户定义的标准的元素进行计数,
 * 该算法返回目标序列中元素 i 的数量, 其中 pred(i)为true 或 value == i;
 * 通常,DifferenceType是 size_t, 但它取决于 InputIterator 的实现;
 * *当想要计算特定值的出现次数时, 使用 std::count;
 * *当想要使用更复杂的谓词进行比较时, 使用 std::count_if;
 * ?DifferenceType std::count([ep], ipt_begin, ipt_end, value);
 * ?DifferenceType std::count_if([ep], ipt_begin, ipt_end, pred);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 InputIterator 对象, 即 ipt_begin/ipt_end, 代表目标序列;
 * 3. value 或一元谓词 pred, 用于评估是否应对目标序列中的元素 x 计数;
 * 
 * *复杂度
 * 线性复杂度, 当没有给出执行策略时,该算法进行distance(ipt_begin, ipt_end) 次比较或pred调用;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::count and std::count_if algorithm\n";
    std::vector<std::string> words{"jelly", "jar", "and", "jam"};

    const auto n_ands = count(words.cbegin(), words.cend(), "and");
    assert(n_ands == 1);

    const auto contains_a = [](const auto &word)
    {
        return word.find('a') != std::string::npos;
    };
    const auto count_if_result = count_if(words.cbegin(), words.cend(), contains_a);
    assert(count_if_result == 3);

    return 0;
}
