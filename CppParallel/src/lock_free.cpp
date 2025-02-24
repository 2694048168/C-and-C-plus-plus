/**
 * @file lock_free.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief An example with of lock-free
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

// --------------------------------------
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
            // Start by reading the current value
            int desired;
            int expected = sink.load();
            do
            {
                // Update the current desired value
                desired = expected + 1;
                // Try CAS until successful
            }
            while (!sink.compare_exchange_strong(expected, desired));
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
