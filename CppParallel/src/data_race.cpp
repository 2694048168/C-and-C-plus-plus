/**
 * @file data_race.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief A parallel program with a data race
 * @version 0.1
 * @date 2025-02-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

// -------------------------------------
int main(int argc, const char *argv[])
{
    // Number of total increments and increments per thread
    const int num_increments        = 1 << 20;
    const int num_threads           = 8;
    const int increments_per_thread = num_increments / num_threads;

    // Incremented by each thread
    volatile int sink = 0;

    // Create a work function to add elements
    auto work = [&]()
    {
        for (int i = 0; i < increments_per_thread; i++)
        {
            // 可以通过反汇编查看生成的指令差异
            // sink = sink + 1;
            // sink += 1;
            ++sink;
        }
    };

    // Spawn threads
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    // Join the threads and check the result
    for (auto &thread : threads)
    {
        if (thread.joinable())
            thread.join();
    }
    assert(sink == num_increments);
    std::cout << "All good done!\n";

    return 0;
}
