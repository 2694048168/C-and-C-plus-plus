/**
 * @file 18_mergeAlgorithms.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <vector>

/**
 * @brief Merging Algorithms
 * 合并算法可以合并两个排序的目标序列,使得输出序列包含两个目标序列的副本并且也是有序的.
 * merge 算法合并两个排序的序列,
 * 该算法将两个目标序列复制到一个输出序列中, 输出序列根据 operator＜ 或 comp(如果提供的话)排序.
 * ?OutputIterator merge([ep], ipt_begin1, ipt_end1, ipt_begin2, ipt_end2, opt_result, [comp]);
 * 1. 可选的 std::execution 执行策略 ep（默认值为 std::execution::seq);
 * 2. 两对 InputIterator, 即 ipt_begin1/ipt_end1 和 ipt_begin2/ipt_end2,代表两个目标序列;
 * 3. OutputIterator, 即 opt_result, 代表输出序列;
 * 4. 确定组成员资格的谓词 pred;
 * *复杂度
 * 线性复杂度, 算法最多进行 N-1 次比较, 其中 N 等于 distance(ipt_begin1,ipt_end1)
 *  + distance(ipt_begin2, ipt_end2).
 * *其他要求
 * 目标序列必须根据 operator＜ 或 comp(如果提供的话)排序.
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "\n======== std::merge algorithm =======\n";
    std::vector<int> numbers1{1, 4, 5};
    std::vector<int> numbers2{2, 3, 3, 6};
    std::vector<int> result;

    std::merge(numbers1.begin(), numbers1.end(), numbers2.begin(), numbers2.end(), back_inserter(result));
    for (const auto &elem : result)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;

    return 0;
}
