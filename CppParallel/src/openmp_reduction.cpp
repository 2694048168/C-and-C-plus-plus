/**
 * @file openmp_reduction.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief A simple example of OpenMP reduction
 * @version 0.1
 * @date 2025-02-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "benchmark/benchmark.h"

#include <omp.h>

#include <random>
#include <vector>

// --------------------------------------
static void baseline(benchmark::State &s)
{
    // Create a random number generator
    std::mt19937 mt(std::random_device{}());

    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Create vectors of random numbers
    const int num_elements = 1 << 20;

    std::vector<float> v_in;
    std::generate_n(std::back_inserter(v_in), num_elements, [&] { return dist(mt); });

    // Timing loop
    for (auto _ : s)
    {
        // Create our variable to accumulate into
        float sink = 0;

        // Run the sum of squares
        for (int i = 0; i < num_elements; i++)
        {
            // Square v_in and add to sink
            benchmark::DoNotOptimize(sink += v_in[i] * v_in[i]);
        }
    }
}

BENCHMARK(baseline)->Unit(benchmark::kMicrosecond)->UseRealTime();

static void baseline_openmp(benchmark::State &s)
{
    // Create a random number generator
    std::mt19937 mt(std::random_device{}());

    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Create vectors of random numbers
    const int num_elements = 1 << 20;

    std::vector<float> v_in;
    std::generate_n(std::back_inserter(v_in), num_elements, [&] { return dist(mt); });

    // Timing loop
    for (auto _ : s)
    {
        // Create our variable to accumulate into
        float sink = 0;

// Run the sum of squares
// Parallelize the for loop
#pragma omp parallel for reduction(+ : sink)
        for (int i = 0; i < num_elements; i++)
        {
            // Square v_in and add to sink
            benchmark::DoNotOptimize(sink += v_in[i] * v_in[i]);
        }
    }
}

BENCHMARK(baseline_openmp)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK_MAIN();
