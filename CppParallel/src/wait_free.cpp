/**
 * @file wait_free.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-24
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

// ------------------------------------
int main(int argc, const char *argv[])
{
    // Set up the number of increments
    // const int iterations       = 1 << 25;
    const int iterations       = 1 << 15;
    const int num_threads      = 8;
    const int items_per_thread = iterations / num_threads;

    // Shared value to increment
    std::atomic<int> sink = 0;

    // Normal work function
    auto work = [&]()
    {
        for (int i = 0; i < items_per_thread; i++)
        {
            // sink++;
            ++sink;
        }
    };

    // Create threads
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    // Wait for threads
    for (auto &thread : threads)
    {
        if (thread.joinable())
            thread.join();
    }

    // Check the results
    std::cout << "Completed " << sink << " iterations\n";

    return 0;
}
