/**
 * @file vector_addition_baseline.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief Baseline vector addition
 * @version 0.1
 * @date 2025-02-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <random>

// ---------------------------------------
int main(int argc, const char *argv[])
{
    // Create some large arrays
    const int num_elements = 1 << 26;

    float *a = new float[num_elements];
    float *b = new float[num_elements];
    float *c = new float[num_elements];

    // Create our random numbers
    std::mt19937 mt(std::random_device{}());

    std::uniform_real_distribution dist(1.0f, 2.0f);

    // Initialize a and b
    std::generate(a, a + num_elements, [&] { return dist(mt); });
    std::generate(b, b + num_elements, [&] { return dist(mt); });

    // Get time before
    auto start = omp_get_wtime();

    // Do vector addition
    for (int i = 0; i < num_elements; i++)
    {
        c[i] = a[i] + b[i];
    }

    // Get time after
    auto end = omp_get_wtime();
    std::cout << end - start << '\n';

    // Free our memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
