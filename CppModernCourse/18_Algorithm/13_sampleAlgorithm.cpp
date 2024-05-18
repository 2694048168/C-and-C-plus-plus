/**
 * @file 13_sampleAlgorithm.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

/**
 * @brief sample(采样)算法产生随机的、稳定的子序列,
 * 该算法从生成序列(population sequence)中抽取 min(pop_end - pop_begin, n) 元素.
 *  有点不寻常的是, 当且仅当 ipt_begin 是前向迭代器时, 该样本将被排序, 它返回输出序列的末端.
 * ?OutputIterator sample([ep], ipt_begin, ipt_end, result, n, urb_generator);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 InputIterator 对象, 即 ipt_begin/ipt_end, 代表生成序列(要采样的序列);
 * 3. 一个 OutputIterator, 即 result, 代表输出序列;
 * 4. 一个 Distance, 即n, 代表要采样的元素的数量;
 * 5. 一个 UniformRandomBitGenerator urb_generator 的 Mersenne Twister 引擎 std::mt19937_64.
 * *复杂度
 * 线性复杂度, 算法的复杂度与 distance(ipt_begin, ipt_end) 成正比.
 * 
 */
const std::string population = "ABCD";
const size_t      n_samples{1'000'000};
std::mt19937_64   urbg;

void sample_length(size_t n)
{
    std::cout << "-- Length " << n << " --\n";
    std::map<std::string, size_t> counts;

    for (size_t i{}; i < n_samples; i++)
    {
        std::string result;
        std::sample(population.begin(), population.end(), std::back_inserter(result), n, urbg);
        counts[result]++;
    }

    for (const auto &[sample, n] : counts)
    // for (const auto &elem : counts)
    {
        const auto percentage = 100 * n / static_cast<double>(n_samples);
        // std::cout << percentage << " '" << elem.first << "'\n";
        std::cout << percentage << " '" << sample << "'\n";
    }
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "\n======== std::sample algorithm ========\n";
    std::cout << std::fixed << std::setprecision(1);
    sample_length(0);
    sample_length(1);
    sample_length(2);
    sample_length(3);
    sample_length(4);

    /**
     * @brief shuffle 算法产生随机排列,
     * 该算法将目标序列随机化, 使这些元素的每个可能的排列都有相等的出现概率.
     * ?void shuffle(rnd_begin, rnd_end, urb_generator);
     * 1. 一对 RandomAccessIterator, 即rnd_begin/rnd_end,代表目标序列;
     * 2. 一个 UniformRandomBitGenerator urb_generator Mersenne Twister 引擎 std::mt19937_64.
     * *复杂度
     * 线性复杂度, 该算法调用 swap 恰好 distance(rnd_begin, rnd_end) - 1 次
     * *其他要求
     * 目标序列的元素必须是可交换的.
     * 
     */
    std::cout << "\n======== std::shuffle algorithm ========\n";
    std::map<std::string, size_t> samples;
    std::cout << std::fixed << std::setprecision(1);

    for (size_t i{}; i < n_samples; i++)
    {
        std::string result{population};
        std::shuffle(result.begin(), result.end(), urbg);
        samples[result]++;
    }

    for (const auto &[sample, n] : samples)
    {
        const auto percentage = 100 * n / static_cast<double>(n_samples);
        std::cout << percentage << " '" << sample << "'\n";
    }

    return 0;
}
