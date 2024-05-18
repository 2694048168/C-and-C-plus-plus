/**
 * @file 20_numericOperations.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>

/**
 * @brief 数值运算 <numeric>
 * C++一些 stdlib 数值运算允许传递一个运算符来自定义行为,
 * ＜functional＞头文件提供了以下类模板, 这些模板通过 operator(T x, T y) 公开各种二元算术运算:
 * ?1. plus＜T＞ 实现了加法 x + y;
 * ?2. minus＜T＞ 实现了减法 x - y;
 * ?3. multiplies＜T＞ 实现了乘法 x * y;
 * ?4. divides＜T＞ 实现了除法 x / y;
 * ?5. modulus＜T＞ 实现了模运算 x % y;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]The plus operator\n";
    std::plus<short> adder;
    assert(3 == adder(1, 2));
    assert(3 == std::plus<short>{}(1, 2));

    /**
     * @brief iota 算法用增量值填充序列,
     * 该算法将从 start 开始的增量值赋值给目标序列.
     * ?void iota(fwd_begin, fwd_end, start);
     * 1. 一对迭代器, 即 fwd_begin/fwd_end, 代表目标序列;
     * 2. 起始值 start;
     * *复杂度
     * 线性复杂度, 算法进行 N 个自增和赋值计算, 其中 N 等 于 distance(fwd_begin,fwd_end).
     * *其他要求
     * 这些对象必须可赋值到 start;
     * 
     */
    std::cout << "[====]The std::iota operator\n";
    std::array<int, 3> easy_as;

    std::iota(easy_as.begin(), easy_as.end(), 1);
    for (const auto &elem : easy_as)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;

    /**
     * @brief accumulate 算法折叠序列(按顺序)
     * *"折叠序列"意味着对序列的元素应用特定的操作, 同时将累积的结果传递给下一个操作.
     * 该算法将 op 应用于 start 和目标序列的第一个元素,
     * 它获取结果和目标序列的下一个元素, 并再次应用 op, 不断以这种方式进行,
     * 直到它访问了目标序列中的每个元素, 算法将目标序列元素和 start 相加,并返回结果.
     * ?T accumulate(ipt_begin, ipt_end, start, [op]);
     * 1. 一对迭代器,即 ipt_begin/ipt_end, 代表目标序列;
     * 2. 起始值 start;
     * 3. 可选的二元运算符 op,默认为 plus;
     * *复杂度
     * 线性复杂度, 算法将 op 应用 N 次, N 等于 distance(ipt_begin, ipt_end);
     * *其他要求
     * 目标序列的元素必须是可复制的;
     * 
     */
    std::cout << "[====]The std::accumulate operator\n";
    std::vector<int> nums{1, 2, 3};

    const auto result1 = std::accumulate(nums.begin(), nums.end(), -1);
    std::cout << "The accumulate result: " << result1 << '\n';

    const auto result2 = std::accumulate(nums.begin(), nums.end(), 2, std::multiplies<>());
    std::cout << "The accumulate result: " << result2 << '\n';

    /**
     * @brief reduce 算法也折叠序列(不一定按顺序),
     * 该算法与 accumulate 相同, 只是它接受一个可选的执行策略参数,并且不保证运算符应用的顺序
     * ?T reduce([ep], ipt_begin, ipt_end, start, [op]); 
     * 1. 可选的 std::execution 执行策略 ep（默认值为 std::execution::seq);
     * 2. 一对迭代器，即 ipt_begin / ipt_end, 代表目标序列;
     * 3. 起始值 start;
     * 4. 可选的二元运算符 op, 默认为 plus;
     * *复杂度
     * 线性复杂度, 算法将 op 应用 N 次, N 等于 distance(ipt_begin, ipt_end).
     * *其他要求
     * 1. 如果省略 ep, 元素必须是可移动的;
     * 2. 如果提供 ep, 元素必须是可复制的;
     * 
     */
    std::cout << "[====]The std::reduce operator\n";
    std::vector<int> nums_{1, 2, 3, 4, 5, 6, 7, 8, 9};

    const auto result3 = std::reduce(nums.begin(), nums.end(), -1);
    std::cout << "The reduce result: " << result3 << '\n';

    const auto result4 = std::reduce(nums.begin(), nums.end(), 2, std::multiplies<>());
    std::cout << "The reduce result: " << result4 << '\n';

    /**
     * @brief inner_product 算法计算两个序列的内积
     * *内积(或点积)是与一对序列相关的标量值
     * 该算法将 op2 应用于目标序列中的每一对相应元素, 并使用 op1 将它们与 start相加
     * ?T inner_product([ep], ipt_begin1, ipt_end1, ipt_begin2, start, [op1], [op2]);
     * 1. 一对迭代器，即 ipt_begin1 / ipt_end1,代表目标序列 1;
     * 2. 迭代器 ipt_begin2,代表目标序列 2;
     * 3. 起始值 start;
     * 4. 两个可选的二元运算符, op1 和 op2, 默认为 plus 和 multiply;
     * *复杂度
     * 线性复杂度, 算法将 op1 和 op2 应用 N 次, 其中 N 等于 distance(ipt_begin1,ipt_end1)
     * *其他要求
     * 元素必须是可复制的
     *
     */
    std::cout << "[====]The std::inner_product operator\n";
    std::vector<int> nums1{1, 2, 3, 4, 5};
    std::vector<int> nums2{1, 0, -1, 0, 1};

    const auto res = std::inner_product(nums1.begin(), nums1.end(), nums2.begin(), 10);
    assert(res == 13);

    /**
     * @brief adjacent_difference 算法生成邻差(即相邻元素的差)
     * !图像梯度计算？
     * *邻差是对每对相邻元素应用某些操作的结果
     * 该算法将输出序列的第一个元素设置为目标序列的第一个元素,
     * 对于后续的每一个元素, 将 op 应用于前一个元素和当前的元素,
     * 并将返回值写入 result 中, 该算法返回输出序列的终点.
     * ?OutputIterator adjacent_difference([ep], ipt_begin, ipt_end, result, [op]);
     * 1. 一对迭代器，即 ipt_begin / ipt_end，代表目标序列;
     * 2. 迭代器 result，代表输出序列;
     * 3. 可选的二元运算符 op，默认为 minus;
     * *复杂度
     * 线性复杂度, 算法将 op 应用 N-1 次,其中 N 等于 distance(ipt_begin, ipt_end)
     * *其他要求
     * 1. 如果省略 ep, 元素必须是可移动的;
     * 2. 如果提供 ep, 元素必须是可复制的;
     * 
     */
    std::cout << "[====]The std::adjacent_difference operator\n";
    std::vector<int> fib{1, 1, 2, 3, 5, 8}, fib_diff;

    std::adjacent_difference(fib.begin(), fib.end(), back_inserter(fib_diff));
    for (const auto &elem : fib)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;

    /**
     * @brief partial_sum 算法生成部分元素的和
     * 该算法设置一个等于目标序列的第一个元素的累加器,
     * 对于目标序列的每个后续元素, 算法将该元素加到累加器,
     * 然后将累加器值写入输出序列. 该算法返回输出序列的终点.
     * ?OutputIterator partial_sum(ipt_begin, ipt_end, result, [op]);
     * 1. 一对迭代器, 即 ipt_begin/ipt_end,代表目标序列;
     * 2. 迭代器 result, 代表输出序列;
     * 3. 可选的二元运算符 op, 默认为 plus;
     * *复杂度
     * 线性复杂度, 算法将 op 应用 N-1 次, 其中 N 等于 distance(ipt_begin, ipt_end)
     * 
     */
    std::cout << "[====]The std::partial_sum operator\n";
    std::vector<int> num{1, 2, 3, 4};
    std::vector<int> result;

    std::partial_sum(num.begin(), num.end(), std::back_inserter(result));
    for (const auto &elem : result)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;

    /**
     * @brief 
     * ?1.（最大）堆操作 ＜algorithm＞ 头文件
     * ?2. 有序区间的集合操作 ＜algorithm＞ 头文件
     * ?3. 其他数值算法 ＜numeric＞ 头文件
     * ?4. 内存操作 ＜memory＞ 头文件
     */

    return 0;
}
