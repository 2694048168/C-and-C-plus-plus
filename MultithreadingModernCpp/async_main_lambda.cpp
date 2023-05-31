/**
 * @file async_main_lambda.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-30
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ .\async_main_lambda.cpp -std=c++17
 * $ g++ .\async_main_lambda.cpp -std=c++17
 * 
 */

#include <stdint.h>

#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

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

    for (uint64_t i = 0; i < num_threads; i++)
    {
        tasks.push_back(std::async(
            [i, step]
            {
                uint64_t r = 0;
                for (uint64_t j = i * step; j < (i + 1) * step; j++)
                {
                    r += j;
                }

                return r;
            }));
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