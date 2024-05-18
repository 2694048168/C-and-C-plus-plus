/**
 * @file 19_extremeValueAlgorithms.cpp
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
#include <vector>

/**
 * @brief Extreme-Value Algorithms 极值算法
 * 极值算法可以确定最小和最大元素, 还可以对元素的最小或最大值进行限制.
 */

// ------------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief min 或 max 算法可以确定序列的极值,
     * 这些算法使用 operator＜ 或 comp, 返回最小(min)或最大(max)对象.
     * minmax 算法以 std::pair 的形式返回两者, first 存储最小值, second 存储最大值.
     * ?T min(obj1, obj2, [comp]);
     * ?T min(init_list, [comp]);
     * ?T max(obj1, obj2, [comp]);
     * ?T max(init_list, [comp]);
     * ?Pair minmax(obj1, obj2, [comp]);
     * ?Pair minmax(init_list, [comp]);
     * 1. 两个对象，obj1 和 obj2;
     * 2. 初始化列表 init_list，代表要比较的对象;
     * 3. 可选的比较函数 comp;
     * *复杂度
     * 固定复杂度或者线性复杂度, 对于使用 obj1 和 obj2 的重载, 正好只进行一次比较;
     * 对于初始化列表, 最多进行 N-1 次比较, 其中 N 是初始化列表的长度;
     * 对于 minmax, 给定一个初始化列表, 复杂度增长到 3N/2.
     * *其他要求
     * 这些元素必须是可以复制构造的,并且可以使用给定的比较法进行比较.
     */
    std::cout << "\n======== std::min and std::max algorithm =======\n";
    using namespace std::literals;
    auto length_compare = [](const auto &x1, const auto &x2)
    {
        return x1.length() < x2.length();
    };

    auto res = std::min("undiscriminativeness"s, "vermin"s, length_compare);
    std::cout << "The min-value :" << res << '\n';

    res = std::max("maxim"s, "ultramaximal"s, length_compare);
    std::cout << "The max-value :" << res << '\n';

    const auto result = std::minmax("minimaxes"s, "maximin"s, length_compare);
    assert(result.first == "maximin");
    assert(result.second == "minimaxes");

    /**
     * @brief min_element 或 max_element 算法确定序列的极值,
     * 算法使用 operator＜ 或 comp, 返回一个指向最小(min_element)
     * 或最大(max_element)对象的迭代器.
     * minmax_element 算法以 std::pair 的形式返回两者, first 为最小值, second 为最大值.
     * ?ForwardIterator min_element([ep], fwd_begin, fwd_end, [comp]);
     * ?ForwardIterator max_element([ep], fwd_begin, fwd_end, [comp]);
     * ?Pair minmax_element([ep], fwd_begin, fwd_end, [comp]);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 ForwardIterator, 即 fwd_begin/fwd_end, 代表目标序列;
     * 3. 可选的比较函数 comp;
     * *复杂度
     * 线性复杂度, 对于 max 和 min, 最多执行 N-1 次比较, 
     * 其中 N 等于 distance(fwd_begin, fwd_end); 对于 minmax 为 3N/2.
     * *其他要求
     * 这些元素必须使用给定的操作来进行比较.
     */
    std::cout << "\n======== std::min_element and std::max_element algorithm =======\n";
    std::vector<std::string> words{"civic", "deed", "kayak", "malayalam"};

    auto iter = std::min_element(words.begin(), words.end(), length_compare);
    std::cout << "The MIN-value: " << *iter << '\n';

    iter = std::max_element(words.begin(), words.end(), length_compare);
    std::cout << "The MAX-value: " << *iter << '\n';

    auto result_ = std::minmax_element(words.begin(), words.end(), length_compare);
    assert(*result_.first == "deed");
    assert(*result_.second == "malayalam");

    /**
     * @brief clamp 算法对一个值进行限定,
     * 该算法使用 operator＜ 或 comp 来确定 obj 是否在从 low 到 high 的范围内,
     * 如果是, 那么算法简单地返回 obj; 否则, 如果 obj 小于low, 则返回 low,
     * 如果 obj 大于 high, 则返回 high;
     * ?T& clamp(obj, low, high, [comp]);
     * 1. 对象 obj;
     * 2. low 和 high 对象;
     * 3. 可选的比较函数 comp;
     * *复杂度
     * 固定复杂度,算法最多比较两次;
     * *其他要求
     * 这些对象必须使用给定的操作来进行比较;
     * 
     */
    std::cout << "\n======== std::clamp algorithm =======\n";
    std::cout << "the result value: " << std::clamp(9000, 0, 100) << '\n';
    std::cout << "the result value: " << std::clamp(-123, 0, 100) << '\n';
    std::cout << "the result value: " << std::clamp(-123, 0, 100) << '\n';

    return 0;
}
