/**
 * @file dot_product_unsafe.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief Dot product implementation with auto-vectorization
 * @version 0.1
 * @date 2025-02-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <algorithm>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

// -------------------------------------
int main(int argc, const char *argv[])
{
    // Create random number generator
    std::mt19937                   mt(123);
    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Create vectors of random numbers
    const int num_elements = 1 << 20;

    std::vector<float> v1;
    std::vector<float> v2;

    std::ranges::generate_n(std::back_inserter(v1), num_elements, [&] { return dist(mt); });
    std::ranges::generate_n(std::back_inserter(v2), num_elements, [&] { return dist(mt); });

    // Perform dot product
    float result = std::transform_reduce(std::execution::unseq, v1.begin(), v1.end(), v2.begin(), 0.0f);

    std::cout << result << '\n';

    return 0;
}
