/**
 * @file dot_product_baseline.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief Vectorized implementation of a dot product
 * @version 0.1
 * @date 2025-02-24
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "benchmark/benchmark.h"

#include <algorithm>
#include <execution>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

// --------------------------------------
static void dp_bench(benchmark::State &s)
{
    // Create a random number generator
    std::mt19937                  mt(std::random_device{}());
    std::uniform_int_distribution dist(1, 10);

    // Create vectors of random numbers
    const int        num_elements = 1 << 15;
    std::vector<int> v1;
    std::vector<int> v2;
    std::ranges::generate_n(std::back_inserter(v1), num_elements, [&] { return dist(mt); });
    std::ranges::generate_n(std::back_inserter(v2), num_elements, [&] { return dist(mt); });

    // Perform dot product
    int *sink = new int;
    for (auto _ : s)
    {
        *sink = std::transform_reduce(std::execution::unseq, v1.begin(), v1.end(), v2.begin(), 0);
    }
}

BENCHMARK(dp_bench)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
