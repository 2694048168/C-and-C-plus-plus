/**
 * @file 01_asynchronousTask.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

/**
 * @brief factorize 函数, 可以找到一个整数的所有因子.
 */
template<typename T>
std::set<T> factorize(T x)
{
    std::set<T> result{1};
    for (T candidate{2}; candidate <= x; candidate++)
    {
        if (x % candidate == 0)
        {
            result.insert(candidate);
            x /= candidate;
            candidate = 1;
        }
    }
    return result;
}

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

// *factor_task 函数,包装了对 factorize 的调用,并返回一个格式良好的信息
std::string factor_task(unsigned long long x)
{
    std::chrono::nanoseconds     elapsed_ns;
    std::set<unsigned long long> factors;
    {
        Stopwatch stopwatch{elapsed_ns};
        factors = factorize<unsigned long long>(x);
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_ns).count();

    std::stringstream ss;
    ss << elapsed_ms << " ms: Factoring " << x << " ( ";
    for (auto factor : factors) ss << factor << " ";
    ss << ")\n";
    return ss.str();
}

std::array<unsigned long long, 6> numbers{9'699'690,     179'426'549,   1'000'000'007,
                                          4'294'967'291, 4'294'967'296, 1'307'674'368'000};

// -----------------------------------
int main(int argc, const char **argv)
{
    std::chrono::nanoseconds elapsed_ns;
    {
        Stopwatch stopwatch{elapsed_ns};
        for (auto number : numbers) std::cout << factor_task(number);
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_ns).count();
    std::cout << elapsed_ms << "ms: total program time\n";

    // *一个使用 factor_task 异步分解六个不同数字的程序
    std::cout << "\n一个使用 factor_task 异步分解六个不同数字的程序\n";
    {
        Stopwatch stopwatch{elapsed_ns};

        std::vector<std::future<std::string>> factor_tasks;
        for (auto number : numbers)
        {
            factor_tasks.emplace_back(async(std::launch::async, factor_task, number));
        }

        for (auto &task : factor_tasks)
        {
            std::cout << task.get();
        }
    }
    const auto elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_ns).count();
    std::cout << elapsed_ms_ << " ms: total program time\n";

    return 0;
}
