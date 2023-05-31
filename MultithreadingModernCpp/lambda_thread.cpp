/**
 * @file lambda_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-30
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ lambda_thread.cpp -std=c++17
 * $ g++ lambda_thread.cpp -std=c++17
 * 
 */

#include <stdint.h>

#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

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
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned int num_threads  = 100;
    const uint64_t     num_elements = 1000 * 1000 * 1000;
    const uint64_t     step         = num_elements / num_threads;

    std::vector<std::thread> threads;

    std::vector<uint64_t> partial_sums(num_threads);

    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.push_back(std::thread(
            [i, &partial_sums, step]
            {
                for (uint64_t j = i * step; j < (i + 1) * step; j++)
                {
                    partial_sums[i] += j;
                }
            }));
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

    return 0;
}