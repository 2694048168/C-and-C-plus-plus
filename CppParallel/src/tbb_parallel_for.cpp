/**
 * @file tbb_parallel_for.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "Stopwatch.hpp"
#include "tbb/parallel_for.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// ---------------------------------------
int main(int argc, const char *argv[])
{
    // Create a random number generator
    std::mt19937 mt(std::random_device{}());

    // Create 1 distribution
    std::uniform_int_distribution bin_dist(20, 30);

    // Calculate the number elements per bin
    int num_work_items = 1 << 18; // 2^18
    // int num_work_items = 1 << 10; // 2^10 = 1024
    std::cout << "The total number of elements is " << num_work_items << '\n';

    // Create work items
    std::vector<int> work_items;
    std::generate_n(std::back_inserter(work_items), num_work_items, [&] { return bin_dist(mt); });
    // for (const auto &elem : work_items)
    for (auto idx{0}; idx < 12; ++idx)
    {
        // std::cout << elem << ' ';
        std::cout << work_items[idx] << ' ';
    }
    std::cout << std::endl;

    long long int sum_single   = 0;
    long long int sum_parallel = 0;
    {
        Stopwatch{"Process in parallel"};

        // Process all elements in a parallel_for loop
        //?Static work distribution with ~equal job length
        // tbb::parallel_for(
        //     tbb::blocked_range<int>(0, num_work_items),
        //     [&](tbb::blocked_range<int> r)
        //     {
        //         for (int i = r.begin(); i < r.end(); ++i)
        //         {
        //             // std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
        //             std::this_thread::sleep_for(std::chrono::microseconds(1));
        //             sum_parallel += work_items[i];
        //         }
        //     },
        //     tbb::static_partitioner());

        // Process all elements in a parallel_for loop
        //?Dynamic work distribution with ~equal job length
        tbb::parallel_for(tbb::blocked_range<int>(0, num_work_items),
                          [&](tbb::blocked_range<int> r)
                          {
                              for (int i = r.begin(); i < r.end(); i++)
                              {
                                  //   std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
                                //   std::this_thread::sleep_for(std::chrono::microseconds(1));
                                  sum_parallel += work_items[i];
                              }
                          });
    }

    {
        Stopwatch{"Process in single"};

        for (const auto &elem : work_items)
        {
            // std::this_thread::sleep_for(std::chrono::microseconds(1));
            sum_single += elem;
        }
    }

    std::cout << "The sum of all elements(parallel) is " << sum_parallel << '\n';
    std::cout << "The sum of all elements(single) is " << sum_single << '\n';

    return 0;
}
