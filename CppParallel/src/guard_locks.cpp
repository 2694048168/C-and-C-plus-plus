/**
 * @file guard_locks.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief A simple example using a mutex
 * @version 0.1
 * @date 2025-02-24
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <random>
#include <thread>

// --------------------------------------
int main(int argc, const char *argv[])
{
    // Random number generate
    std::mt19937 mt(std::random_device{}());

    std::uniform_int_distribution dist(10, 20);

    // Vector of random numbers
    std::list<int> l;
    // std::generate_n(std::back_inserter(l), 1 << 20, [&]() { return dist(mt); });
    std::generate_n(std::back_inserter(l), 1 << 10, [&]() { return dist(mt); });

    // Our lock guarding access to the list
    std::mutex m;

    // Function that removes items from the list in parallel
    auto work = [&]()
    {
        // Run until there are no more items
        while (true)
        {
            // Grab the lock before doing anything
            std::lock_guard<std::mutex> lg(m);

            // Check if there are no more items
            if (l.empty())
                break;

            // Remove the item at the end of the list
            std::cout << l.front() << '\n';
            l.pop_back();
        }
    };

    // Spawn threads
    int num_threads = 8;

    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    return 0;
}
