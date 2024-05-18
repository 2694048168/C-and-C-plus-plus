/**
 * @file 12_removeAlgorithm.cpp
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
#include <string>
#include <vector>

/**
 * @brief remove 算法从一个序列中删除某些元素,
 * 这个算法移动所有 pred 评估为 true 的元素或者等于 value 的元素,
 * 同时保留其余元素的顺序, 它返回一个指向第一个被移动元素的迭代器.
 * 这个迭代器被称为输出序列的逻辑终点.
 * 序列的物理大小保持不变, 调用remove 之后通常还要调用容器的 erase 方法.
 * ?ForwardIterator remove([ep], fwd_begin, fwd_end, value);
 * ?ForwardIterator remove_if([ep], fwd_begin, fwd_end, pred);
 * ?ForwardIterator remove_copy([ep], fwd_begin, fwd_end, result, value);
 * ?ForwardIterator remove_copy_if([ep], fwd_begin, fwd_end, result, pred);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 ForwardIterator, 即 fwd_begin/fwd_end, 代表目标序列;
 * 3. 一个 OutputIterator, 即 result, 代表输出序列(如果复制);
 * 4. 代表要删除的元素的 value;
 * 5. 一元谓词 pred, 它确定元素是否符合删除标准;
 * *复杂度
 * 线性复杂度, 算法调用 pred 或比较 value 恰好 distance(fwd_begin, fwd_end) 次
 * *其他要求
 * 1. 目标序列的元素必须是可移动的;
 * 2. 如果复制, 元素必须是可复制的, 并且目标序列和输出序列不能重叠;
 * 
 */

template<typename T>
void print(std::vector<T> &vec)
{
    std::cout << "The container element: \n";
    for (const auto &elem : vec)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::remove algorithm\n";
    auto is_vowel = [](char x)
    {
        static const std::string vowels{"aeiouAEIOU"};
        return vowels.find(x) != std::string::npos;
    };

    std::string pilgrim
        = "Among the things Billy Pilgrim could not change "
          "were the past, the present, and the future.";

    const auto new_end = std::remove_if(pilgrim.begin(), pilgrim.end(), is_vowel);
    std::cout << pilgrim << " " << *new_end << '\n';

    // TODO:这种 remove(remove_if)和erase 方法的组合被称为 erase-remove 手法被广泛使用
    pilgrim.erase(new_end, pilgrim.end());
    std::cout << pilgrim << '\n';

    /**
     * @brief unique 算法从序列中去除多余的元素,
     * 该算法移动所有 pred 评估为 true 的元素或相等的重复元素,
     * 这样剩余的元素与它们的相邻元素都是唯一的, 并且保留了原来的排序.
     * 它返回一个指向新逻辑终点的迭代器. 与 std::remove 一样, 物理存储空间不会改变.
     * ?ForwardIterator unique([ep], fwd_begin, fwd_end, [pred]);
     * ?ForwardIterator unique_copy([ep], fwd_begin, fwd_end, result, [pred]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 ForwardIterator,即 fwd_begin/fwd_end,代表目标序列;
     * 3. 一个 OutputIterator, 即 result, 代表输出序列(如果复制);
     * 4. 二元谓词 pred, 用于确定两个元素是否相等;
     * *复杂度
     * 线性复杂度, 算法调用 pred 恰好 distance(fwd_begin, fwd_end) - 1 次.
     * *其他要求
     * 1. 目标序列的元素必须是可移动的;
     * 2. 如果复制, 目标序列的元素必须是可复制的, 而且目标序列和输出序列不能重叠;
     * 
     */
    std::cout << "\n[====]std::unique algorithm\n";
    std::string without_walls = "Wallless";

    const auto new_end_ = std::unique(without_walls.begin(), without_walls.end());
    without_walls.erase(new_end_, without_walls.end());
    std::cout << without_walls << '\n';

    /**
     * @brief reverse 算法是将序列的顺序颠倒过来,
     * 该算法通过交换其元素或将其复制到目标序列来逆转一个序列.
     * ?void reverse([ep], bi_begin, bi_end);
     * ?OutputIterator reverse_copy([ep], bi_begin, bi_end, result);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 BidirectionalIterator, 即 bi_begin/bi_end, 代表目标序列;
     * 3. 一个 OutputIterator, 即 result, 代表输出序列(如果复制);
     * *复杂度
     * 线性复杂度, 该算法调用 swap 恰好 distance(bi_begin, bi_end)/2 次.
     * *其他要求
     * 1. 目标序列的元素必须是可交换的;
     * 2. 如果复制, 目标序列的元素必须是可复制的, 而且目标序列和输出序列不能重叠.
     */
    std::cout << "\n[====]std::reverse algorithm\n";
    std::string stinky = "diaper";
    std::cout << "Before reverse: " << stinky << '\n';
    std::reverse(stinky.begin(), stinky.end());
    std::cout << "After reverse: " << stinky << '\n';

    return 0;
}
