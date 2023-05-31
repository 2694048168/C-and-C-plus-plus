/**
 * @file async_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <stdint.h>

#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

/**
 * @brief Get the Range Sum object
 * 
 * @param start 
 * @param end 
 * @return uint64_t 
 */
uint64_t GetRangeSum(const uint64_t start, const uint64_t end)
{
    uint64_t sum = 0;
    for (uint64_t i = start; i < end; ++i)
    {
        sum += i;
    }

    return sum;
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
    const unsigned int num_threads  = 20;
    const uint64_t     num_elements = 1000 * 1000 * 1000;
    const uint64_t     step         = num_elements / num_threads;

    std::vector<std::future<uint64_t>> tasks;

    for (size_t i = 0; i < num_threads; ++i)
    {
        tasks.push_back(std::async(GetRangeSum, i * step, (i + 1) * step));
    }

    uint64_t total = 0;
    for (auto &t : tasks)
    {
        auto p = t.get();
        std::cout << "p: " << p << "\n";
        total += p;
    }

    std::cout << "the total result: " << total << std::endl;

    return 0;
}