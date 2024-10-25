/**
 * @file 06_stop_watch.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/spdlog.h"
#include "spdlog/stopwatch.h"

#include <chrono>
#include <thread>

// ------------------------------------
int main(int argc, const char **argv)
{
    /* 7. StopWatch 计时工具
     * Stopwatch 是 spdlog 提供的一个简单的计时工具, 用于测量代码块执行的时间;
     * 这对性能分析和优化非常有用, 因为它可以帮助开发者了解某段代码的执行时间.
     * 1. 高精度计时: Stopwatch 使用高精度的时钟来测量时间, 可以精确到微秒或更高;
     * 2. 便捷性: Stopwatch 的接口简单易用, 可以直接嵌入代码中用于快速的性能测试;
     * 3. 性能调优: 用于测试代码块执行时间, 找出性能瓶颈;
     * 4. 调试: 在调试时, 可以查看某段代码的执行时间, 以便理解代码的效率;
     */
    // 创建一个 stopwatch 实例
    spdlog::stopwatch stop_watch_monitor;

    // 模拟一些工作负载（睡眠2秒）
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 记录执行时间
    spdlog::info("Elapsed time: {} seconds", stop_watch_monitor.elapsed().count());
    spdlog::info("Elapsed time: {} ms", stop_watch_monitor.elapsed_ms().count());

    /* 代码解释:
    1. 创建 stopwatch 实例: spdlog::stopwatch sw; 
        创建了一个 stopwatch 对象, 计时从对象创建时自动开始.
    2. 模拟工作负载: std::this_thread::sleep_for 函数用于模拟一个耗时 2 秒的操作;
        这个操作可以替换为实际需要测量的代码.
    3. 记录执行时间: 使用 sw.elapsed().count() 获取经过的时间, 并以秒为单位输出日志;
    3. 记录执行时间: 使用 sw.elapsed_ms().count() 获取经过的时间, 并以秒为单位输出日志;

    注意事项:
    * 精度: stopwatch 的精度取决于系统的时钟, 通常可以达到微秒级别,
        但在某些平台上可能受限于硬件或操作系统的计时精度;
    * 多个计时器: 可以在同一个程序中创建多个 stopwatch 实例, 用于测量不同代码块的执行时间;
    */

    return 0;
}
