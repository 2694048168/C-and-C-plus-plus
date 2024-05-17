/**
 * @file 02_forEach.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief for_each
 * for_each 算法将用户定义的函数应用于序列中的每个元素,
 * 该算法将 fn 应用于目标序列的每个元素, 尽管 for_each 被认为是非修改序列操作,
 * 但如果 ipt_begin 是可变的迭代器, fn 可以接受非 const 参数, fn 返回的任何值都会被忽略;
 * 如果省略了 ep, for_each 将返回 fn; 否则, for_each 将返回 void;
 * ?for_each([ep], ipt_begin, ipt_end, fn);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 InputIterator 对象, 即 ipt_begin / ipt_end, 代表目标序列;
 * 3. 一元函数 fn, 它接受来自目标序列的元素;
 *
 * *复杂度
 * 线性复杂度, 算法执行 fn 恰好 distance(ipt_begin, ipt_end) 次;
 * *其他要求
 * 1. 如果省略 ep, fn 必须是可移动的;
 * 2. 如果提供 ep, fn 必须是可复制的;
 * 
 * TODO:https://hackingcpp.com/cpp/std/algorithms.html
 * 
 * *====for_each_n
 * for_each_n 算法将用户定义的函数应用于序列中的每个元素,
 * 该算法将 fn 应用于目标序列的每个元素, 尽管 for_each_n 被认为是非修改序列操作,
 * 但如果 ipt_begin 是可变的迭代器, fn 可以接受非 const 参数.
 * fn 返回的任何值都会被忽略, 它返回 ipt_begin+n;
 * ?InputIterator for_each_n([ep], ipt_begin, n, fn);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. InputIterator ipt_begin 表示目标序列的第一个元素;
 * 3. 表示所需迭代次数的整数 n, 以便表示目标序列的半开半闭区间是 ipt_begin 到 ipt_begin+n
 *    (Size 是 n 的模板类型);
 * 4. 接受来自目标序列的元素的一元函数 fn;
 * *复杂度
 * 线性复杂度, 调用 fn 函数 n 次;
 * *其他要求
 * 1. 如果省略 ep, fn 必须是可移动的;
 * 2. 如果提供 ep, fn 必须是可复制的;
 * 3. n 必须为非负数;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]The for_each algorithm\n";

    std::vector<std::string> words{"David", "Donald", "Doo"};

    size_t number_of_Ds{};

    const auto count_Ds = [&number_of_Ds](const auto &word)
    {
        if (word.empty())
            return;
        if (word[0] == 'D')
            ++number_of_Ds;
    };
    std::for_each(words.cbegin(), words.cend(), count_Ds);

    std::cout << "The 'D' number is " << number_of_Ds << '\n';

    std::cout << "\n[====]The for_each_n algorithm\n";
    std::vector<std::string> words_{"ear", "egg", "elephant"};

    size_t     characters{};
    const auto count_characters = [&characters](const auto &word)
    {
        characters += word.size();
    };
    std::for_each_n(words.cbegin(), words.size(), count_characters);

    std::cout << "The number of character is " << characters << '\n';

    return 0;
}
