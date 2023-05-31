/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-05-29
 *
 * @copyright Copyright (c) 2023
 *
 * $ clang++ main.cpp -std=c++17
 * $ g++ main.cpp -std=c++17
 * $ cl main.cpp /std:c+17 /EHsc
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
 * @brief the multi-threading in Modern C++
 *  A demo for creating two threads to accumulate the range value.
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, const char **argv)
{
    std::cout << "the multi-threading in Modern C++(main thread start)" << std::endl;

    const unsigned int num_threads  = 2;
    const unsigned int num_elements = 1000 * 1000 * 1000;
    const unsigned int step         = num_elements / num_threads;

    std::vector<uint64_t> partial_sums(num_threads);

    std::thread t1(AccumulateRange, std::ref(partial_sums[0]), 0, step);
    std::thread t2(AccumulateRange, std::ref(partial_sums[1]), step, num_threads * step);

    t1.join();
    t2.join();

    uint64_t total = std::accumulate(partial_sums.begin(), partial_sums.end(), uint64_t(0));

    PrintVector(partial_sums);

    std::cout << "the total result: " << total << std::endl;

    std::cout << "the multi-threading in Modern C++(main thread ending)" << std::endl;

    return 0;
}
