/**
 * @file openmp_using.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief An example using OpenMP
 * @version 0.1
 * @date 2025-02-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "benchmark/benchmark.h"

#include <omp.h>

#include <iostream>
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
// Parallelize the for loop
#pragma omp parallel for
        for (int i = 0; i < num_elements; i++)
        {
            // Square v_in and set v_out
            v_out[i] = v_in[i] * v_in[i];
        }
    }

    // std::cout << "Original value: " << v_in[0] << '\n';
    // std::cout << "Square value: " << v_out[0] << '\n';
}

BENCHMARK(baseline)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK_MAIN();
