/**
 * @file functors_thread_unique_ptr.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-30
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ functors_thread_unique_ptr.cpp -std=c++17
 * $ g++ functors_thread_unique_ptr.cpp -std=c++17
 * 
 */

#include <stdint.h>

#include <iostream>
#include <memory>
#include <thread>
#include <vector>

/**
 * @brief TODO
 * 
 */
class AccumulateFunctor
{
public:
    void operator()(uint64_t start, uint64_t end)
    {
        _sum = 0;
        for (auto i = start; i < end; ++i)
        {
            _sum += i;
        }
        std::cout << _sum << "\n";
    }

    // ~AccumulateFunctor(){std::cout << "AccumulateFunctor Destructor." << std::endl;}
    uint64_t _sum;
};

/**
 * @brief TODO
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned int num_core = std::thread::hardware_concurrency();

    const unsigned int num_threads  = num_core;
    const uint64_t     num_elements = 1000 * 1000 * 1000;
    const uint64_t     step         = num_elements / num_threads;

    std::vector<std::thread> threads;

    std::vector<std::unique_ptr<AccumulateFunctor>> functors;

    for (size_t i = 0; i < num_threads; ++i)
    {
        // using unique pointer to avoid memeory leak.
        std::unique_ptr<AccumulateFunctor> functor(new AccumulateFunctor());
        if (functor == nullptr)
        {
            std::cout << "failed to new AccumulateFunctor() pointer" << std::endl;
            continue;
        }
        threads.push_back(std::thread(std::ref(*functor), i * step, (i + 1) * step));

        // ! must be std::move for functor smart pointer
        functors.push_back(std::move(functor));
    }

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    int64_t total = 0;
    for (const auto &pf : functors)
    {
        total += pf->_sum;
    }

    std::cout << "the total result: " << total << std::endl;

    return 0;
}