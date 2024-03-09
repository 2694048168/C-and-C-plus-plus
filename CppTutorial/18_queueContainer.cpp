/**
 * @file 18_queueContainer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习容器之 std::queue 和多线程编程
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

/**
 * @brief 多线程的两个主要用途
 * 1. 保护共享数据(Protecting shared data from concurrent access)
 * 2. 同步并行操作(Synchronizing concurrent operations)
 * 
 * 同步并行操作的三种方法
 * 1. 周期性检查条件(最简单但是最差的实现方法)
 * 2. 使用条件变量(condition variables)
 * 3. 使用futures和promises(理解起来比较复杂, 但有些场景可以很方便)
 * 
 */

/* 2. 使用条件变量(condition variables)
-------------------------------------- */
std::condition_variable g_condVar;
std::mutex              g_mutex;
std::queue<int>         g_queue;

void producer()
{
    int val = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            std::cout << "[====] Producer one value is ready\n";
            g_queue.push(val);
        }
        ++val;

        g_condVar.notify_one();
    }
}

void consumer()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(g_mutex);

        g_condVar.wait(lock, []() { return !g_queue.empty(); });

        auto val = g_queue.front();
        g_queue.pop();
        std::cout << "Received new value: " << val << std::endl;

        lock.unlock();
    }
}

// ===================================
int main(int argc, const char **argv)
{
    std::queue<int> container;

    if (container.empty())
        for (size_t idx = 0; idx < 24; ++idx)
        {
            container.push(idx + 1);
        }

    std::cout << "The First In and First Out (FIFO)\n";
    for (size_t idx = 0; idx < 24; ++idx)
    {
        std::cout << container.front() << ' ';
        container.pop();
    }
    std::cout << std::endl;

    std::cout << "\n============ std::queue 和多线程编程 ============\n";
    std::thread producing_threading(producer);
    std::thread consuming_threading(consumer);

    producing_threading.join();
    consuming_threading.join();

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\18_queueContainer.cpp -std=c++23
// g++ .\18_queueContainer.cpp -std=c++23
