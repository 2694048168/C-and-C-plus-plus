/**
 * @file 06_timingMeasurements.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <cstdio>

/**
 * @brief 测量时间 Timing Measurements
 * 要优化代码,绝对需要准确地测量时间, 可以使用 Chrono 来衡量一系列操作需要多长时间.
 * 这使能够确定特定代码路径实际上是造成某种可观察到的性能问题的原因,
 * 它还使能够为优化工作的进度建立一个客观的衡量标准.
 * 
 * 1. Boost的Timer库在＜boost/timer/timer.hpp＞头文件中包含了 boost::timer::auto_cpu_timer 类,
 *   这是一个 RAII 对象, 它在构造函数中开始计时并在析构函数中停止计时.
 * 2. 可以仅使用 stdlib Chrono 库构建自己的临时 Stopwatch 类.
 * 
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

// -----------------------------------
int main(int argc, const char **argv)
{
    const size_t n = 1'000'000;

    std::chrono::nanoseconds elapsed;

    {
        Stopwatch       stopwatch{elapsed};
        volatile double result{1.23e45};
        for (double i = 1; i < n; i++)
        {
            result /= i;
        }
    }

    // 循环中执行一百万次浮点数除法并计算每次迭代所用的平均时间
    auto time_per_division = elapsed.count() / double{n};
    printf("Took %lld ns total consuming time.\n", elapsed.count());
    printf("Took %g ns per division.\n", time_per_division);

    return 0;
}
