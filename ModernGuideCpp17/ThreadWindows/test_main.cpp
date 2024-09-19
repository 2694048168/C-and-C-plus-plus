/**
 * @file test_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief 通过编程设置CPU亲和性
 * 
 * =====Linux
 * (1) taskset 命令行工具控制整个进程的CPU亲和性;
 * (2) 使用pthread特定的pthread_setafftinity_np函数, 通过设置其亲和性将每个线程固定到单个CPU;
 *
 * =====Windows
 * (1) 使用SetThreadPriority(m_Handle, nPriority);设置线程优先级
 * (2) 使用SetThreadAffinityMask(m_Handle, mask);设置线程CPU亲和性
 * 
 */

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

// -----------------------------------
int main(int argc, const char **argv)
{
    constexpr unsigned num_threads = 4;

    // A mutex ensures orderly access to std::cout from multiplethreads.

    std::mutex iomutex;

    std::vector<std::thread> threads(num_threads);

    for (unsigned i = 0; i < num_threads; ++i)
    {
        threads[i] = std::thread(
            [&iomutex, i]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                while (1)
                {
                    {
                        // Use a lexical scope and lock_guard to safely lock the mutexonly

                        // for the duration of std::cout usage.

                        std::lock_guard<std::mutex> iolock(iomutex);

                        std::cout << "Thread #" << i << ":on CPU " << sched_getcpu() << "\n";
                    }
                    // Simulate important work done by the tread by sleeping for abit...

                    std::this_thread::sleep_for(std::chrono::milliseconds(900));
                }
            });

        // Create a cpu_set_t object representing a set of CPUs. Clear itand mark
        // only CPU i as set.

        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);

        CPU_SET(i, &cpuset);

        int rc = pthread_setaffinity_np(threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);

        if (rc != 0)
        {
            std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        }
    }

    for (auto &t : threads)
    {
        t.join();
    }

    return 0;
}
