/**
 * @file coarse_grained.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief Static work distribution based on large chunks
 * @version 0.1
 * @date 2025-02-20
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <thread>
#include <vector>

// -------------------------------------
int main(int argc, const char *argv[])
{
    // Create a random number generator
    std::mt19937 mt(std::random_device{}());

    // Create 4 different distributions
    std::uniform_int_distribution bin_1(1, 25);
    std::uniform_int_distribution bin_2(26, 50);
    std::uniform_int_distribution bin_3(51, 75);
    std::uniform_int_distribution bin_4(76, 100);

    // Calculate the number elements per bin
    int num_work_items   = 1 << 18;
    int n_bins           = 4;
    int elements_per_bin = num_work_items / n_bins;

    // Create work items
    std::vector<int> work_items;
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_1(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_2(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_3(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin, [&] { return bin_4(mt); });

    // Create a lambda to process a range of items
    auto work = [](std::span<int> items)
    {
        for (const auto item : items)
        {
            // std::this_thread::sleep_for(std::chrono::microseconds(item));
            // std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::cout << "The item value: " << item << '\n';
        }
    };

    // Calculate the number of items per thread (assume this equally divides)
    int num_threads      = 8;
    int items_per_thread = num_work_items / num_threads;

    // Spawn threads (join in destructor of jthread)
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        int start = i * items_per_thread;
        threads.emplace_back(work, std::span(work_items.begin() + start, items_per_thread));
    }

    return 0;
}
