/**
 * @file vector_threads.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-29
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ vector_threads.cpp -std=c++17
 * $ g++ vector_threads.cpp -std=c++17
 * 
 */

#include <stdint.h>

#include <functional>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

/**
 * @brief TODO
 * 
 * @tparam T 
 * @param input 
 */
template<typename T>
void PrintVector(std::vector<T> input_vect)
{
    std::cout << "{ ";
    unsigned int count = 0;
    for (const auto &elem : input_vect)
    {
        ++count;
        std::cout << elem;
        if (count < input_vect.size())
            std::cout << ", ";
    }
    std::cout << " }\n";
}

/**
 * @brief TODO
 * 
 * @param sum 
 * @param start 
 * @param end 
 */
void AccumulateRange(uint64_t &sum, const uint64_t start, const uint64_t end)
{
    sum = 0;
    for (uint64_t i = start; i < end; i++)
    {
        sum += i;
    }
}

/**
 * @brief TODO
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "the multi-threading in Modern C++(main thread start)" << std::endl;

    const unsigned int num_threads{1000};
    const uint64_t     num_elements{1000 * 1000 * 1000};
    const uint64_t     step = num_elements / num_threads;

    std::vector<std::thread> threads;
    std::vector<uint64_t>    partial_sums(num_threads);

    for (uint64_t i = 0; i < num_threads; ++i)
    {
        threads.push_back(std::thread(AccumulateRange, std::ref(partial_sums[i]), i * step, (i + 1) * step));
    }

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    uint64_t total = std::accumulate(partial_sums.begin(), partial_sums.end(), uint64_t(0));

    PrintVector(partial_sums);

    std::cout << "the total result: " << total << std::endl;

    std::cout << "the multi-threading in Modern C++(main thread ending)" << std::endl;

    unsigned int num_core = std::thread::hardware_concurrency();
    std::cout << "\nthe number of cores on this HOST: " << num_core << std::endl;

    return 0;
}