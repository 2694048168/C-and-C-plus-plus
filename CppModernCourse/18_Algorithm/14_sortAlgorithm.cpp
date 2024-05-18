/**
 * @file 14_sortAlgorithm.cpp
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
#include <string>

/**
 * @brief 排序及相关操作 Sorting and Related Operations
 * *排序操作是一种以某种方式重新排序元素的算法.
 * *每种排序算法都有两个版本:一种接受比较运算符函数对象,另一种使用 operator＜.
 * 比较运算符是一种函数对象, 可以使用两个要比较的对象进行调用;
 * 如果第一个参数小于第二个参数, 它返回 true; 否则, 它返回 false;
 * * x ＜ y 的排序解释是 x 被排在 y 之前.
 * ?operator＜ 是一个有效的比较运算符
 * ?比较运算符必须具有传递性
 * 
 */

enum class CharCategory
{
    Ascender,
    Normal,
    Descender
};

CharCategory categorize(char x)
{
    switch (x)
    {
    case 'g':
    case 'j':
    case 'p':
    case 'q':
    case 'y':
        return CharCategory::Descender;
    case 'b':
    case 'd':
    case 'f':
    case 'h':
    case 'k':
    case 'l':
    case 't':
        return CharCategory::Ascender;
    }
    return CharCategory::Normal;
}

bool ascension_compare(char x, char y)
{
    return categorize(x) < categorize(y);
}

// ------------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief sort(排序)算法对序列进行排序(非稳定排序)
     * !稳定的排序算法保留了相等元素的相对顺序,而不稳定的排序算法可能会重排它们.
     * ?void sort([ep], rnd_begin, rnd_end, [comp]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 RandomAccessIterator, 即 rnd_begin/rnd_end, 代表目标序列;
     * 3. 一个可选的比较运算符 comp;
     * *复杂度
     * 拟线性复杂度, O(NlogN), N 为 distance(rnd_begin, rnd_end)
     * *其他要求
     * 目标序列的元素必须是可交换的、可移动构造的，以及可移动赋值的.
     * 
     */
    std::cout << "\n========= std::sort algorithm ========\n";
    std::string goat_grass{"spoilage"};
    std::sort(goat_grass.begin(), goat_grass.end());
    assert(goat_grass == "aegilops");

    /**
     * @brief stable_sort 算法对序列进行稳定的排序,
     * 该算法对目标序列进行适当的排序, 相等的元素保留原来的顺序
     * ?void stable_sort([ep], rnd_begin, rnd_end, [comp]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 RandomAccessIterator, 即 rnd_begin/rnd_end, 代表目标序列;
     * 3. 可选的比较运算符 comp;
     * *复杂度
     * 多项式对数线性复杂度 O(N log2 N), N 等 于 distance(rnd_begin, rnd_end);
     * 当额外内存可用时,复杂度可以降低到拟线性复杂度.
     * *其他要求
     * 目标序列的元素必须是可交换的、可移动构造的，以及可移动赋值的.
     */
    std::cout << "\n========= std::stable_sort algorithm ========\n";
    std::string word{"outgrin"};
    std::stable_sort(word.begin(), word.end(), ascension_compare);
    assert(word == "touring");

    return 0;
}
