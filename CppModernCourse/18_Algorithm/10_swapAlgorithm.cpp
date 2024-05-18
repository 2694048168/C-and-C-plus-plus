/**
 * @file 10_swapAlgorithm.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

/**
 * @brief swap_ranges
 * swap_ranges 算法将元素从一个序列交换到另一个序列,
 * 该算法对序列 1 和序列 2的每个元素调用 swap, 并返回接收序列的结束迭代器.
 * !有责任确保目标序列表示的序列至少与源序列具有一样多的元素.
 * ?OutputIterator swap_ranges([ep], ipt_begin1, ipt_end1, ipt_begin2);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 ForwardIterator, 即 ipt_begin1/ipt_end1, 代表序列 1;
 * 3. 一个 ForwardIterator, 即 ipt_begin2, 代表序列 2 的开头;
 * *复杂度
 * 线性复杂度, 该算法恰好调用 swap distance(ipt_begin1, ipt_end1) 次;
 * *其他要求
 * 每个序列中包含的元素必须是可交换的;
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::swap_ranges algorithm\n";
    std::vector<std::string> words1{"The", "king", "is", "dead."};
    std::vector<std::string> words2{"Long", "live", "the", "king."};

    std::swap_ranges(words1.begin(), words1.end(), words2.begin());

    auto print = [&](std::vector<std::string> &vec)
    {
        std::cout << "The copy operators: \n";
        for (const auto &elem : vec)
        {
            std::cout << elem << ' ';
        }
        std::cout << std::endl;
    };
    print(words1);
    print(words2);

    /**
     * @brief transform 算法修改一个序列的元素并将它们写入另一个序列,
     * 该算法在目标序列的每个元素上调用 unary_op 并将其输出到输出序列中,
     * 或者在每个目标序列的相应元素上调用 binary_op.
     * ?OutputIterator transform([ep], ipt_begin1, ipt_end1, result, unary_op);
     * ?OutputIterator transform([ep], ipt_begin1, ipt_end1, ipt_begin2, result, binary_op);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象, 即 ipt_begin1/ipt_end1, 代表目标序列;
     * 3. 可选的 InputIterator, 即 ipt_begin2, 代表第二个目标序列,
     *    必须确保第二个目标序列的元素数量至少与第一个目标序列相同;
     * 4. 一个 OutputIterator, 即 result, 代表输出序列的开头;
     * 5. 一个单项操作 unary_op, 它将目标序列的元素转换为输出序列的元素,
     *   如果提供两个目标序列, 就需要提供一个二元操作 binary_op, 
     *   它从每个目标序列中接受一个元素, 并将每个元素转换为输出序列的一个元素;
     * *复杂度
     * 线性复杂度, 该算法在目标序列上调用unary_op或binary_op恰好distance(ipt_begin1, ipt_end1)次
     * 
     */
    std::cout << "[====]std::transform algorithm\n";
    std::vector<std::string> words3{"farewell", "hello", "farewell", "hello"};
    std::vector<std::string> result3;

    std::transform(words3.begin(), words3.end(), std::back_inserter(result3),
                   [](std::string &x)
                   {
                       for (size_t idx{0}; idx < x.size(); ++idx)
                       {
                           x[idx] = std::toupper(x[idx]);
                       }
                       return x;
                   });

    print(result3);

    std::vector<std::string> words4{"light", "human", "bro", "quantum"};
    std::vector<std::string> words5{"radar", "robot", "pony", "bit"};
    std::vector<std::string> result5;

    auto portmantize = [](const auto &x, const auto &y)
    {
        const auto  x_letters = std::min(size_t{2}, x.size());
        std::string result{x.begin(), x.begin() + x_letters};
        const auto  y_letters = std::min(size_t{3}, y.size());
        result.insert(result.end(), y.end() - y_letters, y.end());
        return result;
    };

    std::transform(words4.begin(), words4.end(), words5.begin(), back_inserter(result5), portmantize);
    print(result5);

    return 0;
}
