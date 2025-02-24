/**
 * @file concurrent_queue.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief A simple example using a concurrent queue
 * @version 0.1
 * @date 2025-02-24
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "tbb/concurrent_queue.h"

#include <iostream>
#include <random>
#include <thread>
#include <vector>

// ----------------------------------------
int main(int argc, const char *argv[])
{
    // Number of elements to add to the queue
    // const int num_elements = 1 << 25;
    const int num_elements = 1 << 10;

    // Divide the work by threads
    const int num_threads         = 8;
    const int elements_per_thread = num_elements / num_threads;

    // Create a concurrent queue
    tbb::concurrent_queue<int> queue;

    // Create a random number generator
    std::mt19937                  mt(std::random_device{}());
    std::uniform_int_distribution dist(1, 100);

    // Create a work function to add elements
    auto work = [&]()
    {
        for (int i = 0; i < elements_per_thread; i++)
        {
            queue.push(dist(mt));
            std::cout << dist(mt) << '\n';
        }
    };

    // Spawn threads
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    return 0;
}
