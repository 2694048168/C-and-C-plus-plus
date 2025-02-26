/**
 * @file tbb_using.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief An example parallelized using TBB directly
 * @version 0.1
 * @date 2025-02-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "benchmark/benchmark.h"
#include "tbb/parallel_for.h"

#include <execution>
#include <random>
#include <vector>

// -----------------------------------------
static void baseline(benchmark::State &s)
{
    // Create a random number generator
    std::mt19937 mt(std::random_device{}());

    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Create vectors of random numbers
    const int num_elements = 1 << 20;

    std::vector<float> v_in;
    std::generate_n(std::back_inserter(v_in), num_elements, [&] { return dist(mt); });

    // Output vector is just 0s
    std::vector<float> v_out(num_elements);

    // Timing loop
    for (auto _ : s)
    {
        // Parallelize using TBB parallel_for loop
        tbb::parallel_for(tbb::blocked_range<int>(0, num_elements),
                          [&](tbb::blocked_range<int> r)
                          {
                              // Square v_in and set v_out
                              for (int i = r.begin(); i < r.end(); i++)
                              {
                                  v_out[i] = v_in[i] * v_in[i];
                              }
                          });
    }
}

BENCHMARK(baseline)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK_MAIN();
