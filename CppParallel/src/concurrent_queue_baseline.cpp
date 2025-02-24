/**
 * @file concurrent_queue_baseline.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-24
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

// --------------------------------------
int main(int argc, const char *argv[])
{
    // Number of elements to add to the queue
    const int num_elements = 1 << 25;

    // Divide the work by threads
    const int num_threads         = 8;
    const int elements_per_thread = num_elements / num_threads;

    // Create a queue
    std::queue<int> queue;

    // Create a mutex to guard the queue
    std::mutex m;

    // Create a random number generator
    std::mt19937                  mt(std::random_device{}());
    std::uniform_int_distribution dist(1, 100);

    // Create a work function to add elements
    auto work = [&]()
    {
        for (int i = 0; i < elements_per_thread; i++)
        {
            auto                        val = dist(mt);
            std::lock_guard<std::mutex> lg(m);
            queue.push(val);
            std::cout << val << '\n';
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
