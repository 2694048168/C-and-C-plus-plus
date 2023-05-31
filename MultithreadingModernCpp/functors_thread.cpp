/**
 * @file functors_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-29
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ functors_thread.cpp -std=c++17
 * $ g++ functors_thread.cpp -std=c++17
 * 
 */

#include <stdint.h>

#include <functional>
#include <iostream>
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

    std::vector<AccumulateFunctor *> functors;

    for (size_t i = 0; i < num_threads; ++i)
    {
        AccumulateFunctor *functor = new AccumulateFunctor();
        if (functor == nullptr)
        {
            std::cout << "failed to new AccumulateFunctor() pointer" << std::endl;
            continue;
        }
        threads.push_back(std::thread(std::ref(*functor), i * step, (i + 1) * step));

        functors.push_back(functor);
        // TODO delete the functor pointer? memory leak
        // ! delete the 'functor' here will cause incorrect result.
        // delete functor;
    }

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    int64_t total = 0;
    for (auto pf : functors)
    {
        total += pf->_sum;
    }

    std::cout << "the total result: " << total << std::endl;

    return 0;
}