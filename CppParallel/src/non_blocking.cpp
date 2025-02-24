/**
 * @file non_blocking.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief An example with non-blocking
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

// -------------------------------------
int main(int argc, const char *argv[])
{
    // Set up the number of increments
    // const int iterations  = 1 << 15;
    const int iterations  = 1 << 10;
    const int num_threads = 8;

    // Shared value to increment
    std::atomic<int> sink = 0;

    // Normal work function
    auto work = [&]()
    {
        while (true)
        {
            // Start by reading the current value
            int desired;
            int expected = sink.load();
            do
            {
                // Update the current desired value
                if (expected == iterations)
                    return;
                desired = expected + 1;
                // Try CAS until successful
            }
            while (!sink.compare_exchange_strong(expected, desired));
        }
    };

    // Slow work function
    auto slow_work = [&]()
    {
        while (true)
        {
            // Start by reading the current value
            int desired;
            int expected = sink.load();
            do
            {
                // Update the current desired value
                if (expected == iterations)
                    return;
                desired = expected + 1;
                // Try CAS until successful
            }
            while (!sink.compare_exchange_strong(expected, desired));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    };

    // Create threads
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads - 1; i++)
    {
        threads.emplace_back(work);
    }
    threads.emplace_back(slow_work);

    // Wait for threads
    for (auto &thread : threads) thread.join();

    // Check the results
    std::cout << "Completed " << sink << " iterations\n";

    return 0;
}
