/**
 * @file 08_mutatingSequence.cpp
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
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

/**
 * @brief 可变序列操作 Mutating Sequence Operations
 * ?可变序列操作是一种在序列上进行计算的算法, 允许以某种方式修改序列.
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief copy 算法将一个序列复制到另一个序列,
     * 该算法将目标序列复制到 result 中并返回接收序列的结束迭代器.
     * !有责任确保该 result 具有足够空间来存储目标序列.
     * ?OutputIterator copy([ep], ipt_begin, ipt_end, result);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin/ipt_end, 代表目标序列;
     * 3. 一个 OutputIterator, 即 result, 它接收复制的序列;
     * *复杂度
     * 线性复杂度, 该算法从目标序列复制元素 distance(ipt_begin, ipt_end) 次.
     * *其他要求
     * 除非操作是向左复制, 否则序列 1 和 2 不得重叠;
     * 
     */
    std::cout << "\n[====]std::copy algorithm\n";
    std::vector<std::string> words1{"and", "prosper"};
    std::vector<std::string> words2{"Live", "long"};

    std::copy(words1.cbegin(), words1.cend(), std::back_inserter(words2));
    // assert(words2 == std::vector<std::string>{"Live", "long", "and", "prosper"});
    auto print = [&](std::vector<std::string> &vec)
    {
        std::cout << "The copy operators: \n";
        for (const auto &elem : vec)
        {
            std::cout << elem << ' ';
        }
        std::cout << std::endl;
    };
    print(words2);

    /**
     * @brief copy_n 算法将一个序列复制到另一个序列,
     * 该算法将目标序列复制到 result 中并返回接收序列的结束迭代器.
     * !有责任确保 result 具有足够空间来存储目标序列，并且 n 代表目标序列的正确长度.
     * ?OutputIterator copy_n([ep], ipt_begin, n, result);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一个开始迭代器, 即 ipt_begin, 代表目标序列的开头;
     * 3. 目标序列的大小 n;
     * 4. 一个接收复制的序列的 OutputIterator result;
     * 
     * *复杂度
     * 线性复杂度, 该算法从目标序列复制元素 distance(ipt_begin, ipt_end) 次;
     * *其他要求
     * 序列 1 和序列 2 不能包含相同的对象, 除非是向左复制的操作;
     * 
     */
    std::cout << "\n[====]std::copy_n algorithm\n";
    std::vector<std::string> words3{"on", "the", "wind"};
    std::vector<std::string> words4{"I'm", "a", "leaf"};

    std::copy_n(words3.cbegin(), words3.size(), back_inserter(words4));
    print(words4);

    /**
     * @brief copy_backward 算法将一个序列反向复制到另一个序列中,
     * 该算法将序列 1 复制到序列 2 中并返回接收序列的结束迭代器.
     * 元素向后复制, 但会以原始顺序出现在目标序列中.
     * !有责任确保序列 1 有足够空间存储序列 2.
     * ?OutputIterator copy_backward([ep], ipt_begin1, ipt_end1, ipt_end2);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin1/ipt_end1, 代表序列 1;
     * 3. 一个 InputIterator, 即 ipt_end2, 代表序列 2 的结尾;
     * *复杂度
     * 线性复杂度, 该算法从目标序列复制元素 distance(ipt_begin1, ipt_end1) 次;
     * *其他要求
     * 序列 1 和序列 2 不得重叠.
     * 
     */
    std::cout << "\n[====]std::copy_backward algorithm\n";
    std::vector<std::string> words5{"A", "man", "a", "plan", "a", "bran", "muffin"};
    std::vector<std::string> words6{"a", "canal", "Panama"};

    const auto result = std::copy_backward(words6.cbegin(), words6.cend(), words5.end());
    assert(result == words5.end());
    print(words5);

    return 0;
}
