/**
 * @file 07_ParallelAlgorithms.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

/**
 * @brief C++ stdlib 的算法, 许多算法都需要一个可选的第一个参数,
 * 该参数称为执行策略参数, 由 std::execution 值表示:
 * *在支持的环境中, 它有三个可能的值: seq,par,par_unseq. 如果选择后两个选项,则表示想并行执行该算法
 * 
 */

/**
 * @brief 测量时间 Timing Measurements
 * 要优化代码,绝对需要准确地测量时间, 可以使用 Chrono 来衡量一系列操作需要多长时间.
 * 这使能够确定特定代码路径实际上是造成某种可观察到的性能问题的原因,
 * 它还使能够为优化工作的进度建立一个客观的衡量标准.
 */
struct Stopwatch
{
    Stopwatch(std::chrono::nanoseconds &result)
        : m_result{result}
        , m_start{std::chrono::high_resolution_clock::now()}
    {
    }

    ~Stopwatch()
    {
        m_result = std::chrono::high_resolution_clock::now() - m_start;
    }

private:
    std::chrono::nanoseconds &m_result;

    const std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

std::vector<long> make_random_vector()
{
    std::vector<long> numbers(1'000'000'000);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::mt19937_64 urng{121216};
    std::shuffle(numbers.begin(), numbers.end(), urng);

    return numbers;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "Constructing random vectors...\n";
    auto numbers_a = make_random_vector();
    auto numbers_b{numbers_a};

    std::chrono::nanoseconds time_to_sort;
    std::cout << " " << numbers_a.size() << " elements.\n";
    std::cout << "Sorting with execution::seq...";
    {
        Stopwatch stopwatch{time_to_sort};
        std::sort(std::execution::seq, numbers_a.begin(), numbers_a.end());
    }
    std::cout << " took " << time_to_sort.count() / 1.0E9 << " sec.\n";

    std::cout << "Sorting with execution::par...";
    {
        Stopwatch stopwatch{time_to_sort};
        std::sort(std::execution::par, numbers_b.begin(), numbers_b.end());
    }
    std::cout << " took " << time_to_sort.count() / 1.0E9 << " sec.\n";

    // !并行算法不是魔法 Parallel Algorithms Are Not Magic
    std::cout << "\n===== Parallel Algorithms Are Not Magic =====\n";
    std::vector<long> numbers{1'000'000}, squares{1'000'000};
    std::iota(numbers.begin(), numbers.end(), 0);
    size_t n_transformed{};
    std::transform(std::execution::par, numbers.begin(), numbers.end(), squares.begin(),
                   [&n_transformed](const auto x)
                   {
                       ++n_transformed;
                       return x * x;
                   });
    std::cout << "n_transformed: " << n_transformed << std::endl;

    return 0;
}
