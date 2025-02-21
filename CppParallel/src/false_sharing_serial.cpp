/**
 * @file false_sharing_serial.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief A serial execution benchmark
 * @version 0.1
 * @date 2025-02-21
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

// ---------------------------------------
int main(int argc, const char *argv[])
{
    // Number of total iterations to run
    const int num_iterations = 1 << 27;

    // Number of threads to spawn
    const int num_threads = 1;

    // Atomic integer to increment
    std::atomic<int> counter = 0;

    // Lambda for our work
    auto work = [&]()
    {
        for (int i = 0; i < num_iterations; i++)
        {
            counter++;
        }
    };

    // Spawn threads
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    for (int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }

    std::cout << "The total counter " << counter << '\n';

    return 0;
}
