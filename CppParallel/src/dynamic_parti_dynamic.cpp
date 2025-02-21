/**
 * @file dynamic_parti_dynamic.cpp
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

// ---------------------------------------
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
    int num_work_items = 1 << 18;

    // Number of threads to spawn
    int num_threads = std::thread::hardware_concurrency();

    // Create work items
    std::vector<int> work_items;
    work_items.reserve(num_work_items);
    for (int i = 0; i < num_work_items; i += num_threads)
    {
        // Threads 0/1 get all the data from bin 1
        work_items.push_back(bin_1(mt));
        work_items.push_back(bin_1(mt));

        // Threads 2/3 get all the data from bin 2
        work_items.push_back(bin_2(mt));
        work_items.push_back(bin_2(mt));

        // Threads 4/5 get all the data from bin 3
        work_items.push_back(bin_3(mt));
        work_items.push_back(bin_3(mt));

        // Threads 6/7 get all the data from bin 4
        work_items.push_back(bin_4(mt));
        work_items.push_back(bin_4(mt));
    }

    // Create an atomic to keep track of the next work item
    std::atomic<int> index = 0;

    // Each thread grabs a new element each iteration (in no set order)
    auto work = [&]()
    {
        for (int i = index.fetch_add(1); i < num_work_items; i = index.fetch_add(1))
        {
            // std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::cout << "The task parallel value " << work_items[i] << '\n';
        }
    };

    // Spawn threads (join in destructor of jthread)
    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(work);
    }

    return 0;
}
