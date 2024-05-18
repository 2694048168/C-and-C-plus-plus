/**
 * @file 11_replaceAlgorithm.cpp
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
 * @brief replace 算法用一些新的元素替换序列中的某些元素,
 * 该算法搜索目标序列元素 x 这些元素要么满足 x == old_ref,
 * 要么满足 pred(x) == true, 并将它们赋值为new_ref.
 * ?void replace([ep], fwd_begin, fwd_end, old_ref, new_ref);
 * ?void replace_if([ep], fwd_begin, fwd_end, pred, new_ref);
 * ?void replace_copy([ep], fwd_begin, fwd_end, result, old_ref, new_ref);
 * ?void replace_copy_if([ep], fwd_begin, fwd_end, result, pred, new_ref);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 ForwardIterator, 即 fwd_begin/fwd_end, 代表目标序列;
 * 3. 一个 OutputIterator, 即 result, 代表输出序列的开头;
 * 4. 一个 old const 引用, 代表要找的元素;
 * 5. 一个一元谓词 pred, 它确定元素是否符合替换标准;
 * 6. 一个 new_ref const 引用, 代表要替换的元素;
 * *复杂度
 * 线性复杂度, 该算法调用 pred 恰好 distance(fwd_begin, fwd_end)次;
 * *其他要求
 * 每个序列中包含的元素必须与 old_ref 具有可比性, 并且可以赋值为 new_ref;
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
    std::cout << "[====]std::replace algorithm\n";
    using namespace std::literals;
    std::vector<std::string> words1{"There", "is", "no", "try"};

    std::replace(words1.begin(), words1.end(), "try"sv, "spoon"sv);
    print(words1);

    const std::vector<std::string> words2{"There", "is", "no", "spoon"};
    std::vector<std::string>       words3{"There", "is", "no", "spoon"};

    auto has_two_os = [](const auto &x)
    {
        return count(x.begin(), x.end(), 'o') == 2;
    };
    std::replace_copy_if(words2.begin(), words2.end(), words3.begin(), has_two_os, "try"sv);
    print(words3);

    /**
     * @brief fill 算法用一些值填充序列,
     * 该算法将一个值写入目标序列的每个元素, fill_n 函数返回 opt_begin+n.
     * ?void fill([ep], fwd_begin, fwd_end, value);
     * ?OutputIterator fill_n([ep], opt_begin, n, value);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一个 ForwardIterator, 即 fwd_begin, 代表目标序列的开头;
     * 3. 一个 ForwardIterator, 即 fwd_end, 代表序列的结尾;
     * 4. 一个 Size, 即 n,代表元素的数量;
     * 5. 写入目标序列中每个元素的 value;
     * *复杂度
     * 线性复杂度, 该算法赋值 value 恰好 distance(fwd_begin, fwd_end) 次或n 次.
     * *其他要求
     * 1. value 参数必须可以写入序列中;
     * 2. Size 类型的对象必须可以转换为整数类型;
     * 
     */
    std::cout << "\n[====]std::fill algorithm\n";
    std::vector<std::string> answer1(6);
    std::fill(answer1.begin(), answer1.end(), "police");
    print(answer1);

    std::vector<std::string> answer2;
    std::fill_n(std::back_inserter(answer2), 6, "police");
    print(answer2);

    /**
     * @brief generate 算法通过调用函数对象来填充序列,
     * 该算法调用 generator 并将结果分配给目标序列, generate_n 函数返回 opt_begin+n.
     * ?void generate([ep], fwd_begin, fwd_end, generator);
     * ?OutputIterator generate_n([ep], opt_begin, n, generator);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq)
     * 2. 一个 ForwardIterator, 即 fwd_begin, 代表目标序列的开头;
     * 3. 一个 ForwardIterator, 即 fwd_end, 代表序列的结尾;
     * 4. 一个 Size, 即 n,代表元素的数量;
     * 5. 一个 generator, 当不带参数调用时产生一个写入目标序列的元素;
     * *复杂度
     * 线性复杂度, 该算法调用 generator 恰好 distance(fwd_begin, fwd_end)次或 n 次.
     * *其他要求
     * 1. value 参数必须可以写入序列中;
     * 2. Size 类型的对象必须可以转换为整数类型;
     *
     */
    std::cout << "\n[====]std::generate algorithm\n";
    auto i{1};
    auto pow_of_2 = [&i]()
    {
        const auto tmp = i;
        i *= 2;
        return tmp;
    };

    std::vector<int> series1(6);
    std::generate(series1.begin(), series1.end(), pow_of_2);
    print(series1);

    std::vector<int> series2;
    std::generate_n(std::back_inserter(series2), 6, pow_of_2);
    print(series2);

    return 0;
}
