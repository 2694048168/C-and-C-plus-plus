/**
 * @file work_distribution_uneven_job_length.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "Stopwatch.hpp"
#include "tbb/parallel_for.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>
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

    // Process all elements in a parallel_for loop
    // Static work distribution with uneven job length
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_work_items),
        [&](tbb::blocked_range<int> r)
        {
            for (int i = r.begin(); i < r.end(); i++)
            {
                // std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
                // std::this_thread::sleep_for(std::chrono::microseconds(i));
                // std::this_thread::sleep_for(std::chrono::microseconds(1));
                std::cout << std::to_string(i) + " Static work distribution with uneven job length\n";
            }
        },
        tbb::static_partitioner());

    // Dynamic work distribution with uneven job length
    tbb::parallel_for(tbb::blocked_range<int>(0, num_work_items),
                      [&](tbb::blocked_range<int> r)
                      {
                          for (int i = r.begin(); i < r.end(); i++)
                          {
                              //   std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
                              std::cout << std::to_string(i) + " Dynamic work distribution with uneven job length\n";
                          }
                      });

    return 0;
}
