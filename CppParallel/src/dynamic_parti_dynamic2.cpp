/**
 * @file dynamic_parti_dynamic2.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief Dynamic work distribution based using an atomic index
 * @version 0.1
 * @date 2025-02-21
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <thread>
#include <vector>

// ------------------------------------
int main(int argc, const char *argv[])
{
    // Create a random number generator
    std::mt19937 mt(std::random_device{}());

    // Create 4 different distributions
    std::uniform_int_distribution bin_1(1, 25);
    std::uniform_int_distribution bin_2(26, 50);
    std::uniform_int_distribution bin_3(51, 75);
    std::uniform_int_distribution bin_4(76, 100);

    // Number of elements to process
    int num_work_items   = 1 << 18;
    int n_bins           = 4;
    int elements_per_bin = num_work_items / n_bins;

    // Create work items
    std::vector<int> work_items;
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_1(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_2(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_3(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_4(mt); });
    work_items.reserve(num_work_items);

    // Create an atomic to keep track of the next work item
    std::atomic<int> index = 0;

    // Each thread grabs a new element each iteration (in no set order)
    auto work = [&]()
    {
        for (int i = index.fetch_add(1); i < num_work_items; i = index.fetch_add(1))
        {
            // std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
            // std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::cout << "The task parallel value " << work_items[i] << '\n';
        }
    };

    // Number of threads to spawn
    int num_threads = 8;

    // Spawn threads (join in destructor of jthread)
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    return 0;
}
