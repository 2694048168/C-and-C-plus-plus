#ifndef __RANGE_ACCUMULATOR_HPP__
#define __RANGE_ACCUMULATOR_HPP__

#include "benchmark/benchmark.h"

#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// A class for comparison of lockless and lockfull multithreading
class RangeAccumulator
{
public:
    static void Init()
    {
        _mutex_sum  = 0;
        _atomic_sum = 0;
    }

    // Accumulates values from low to high
    static void AtomicAccumulator(unsigned long low, unsigned long high)
    {
        for (unsigned long i = low; i < high; i++)
        {
            benchmark::DoNotOptimize(_atomic_sum += i);
        }
    }

    //-----------------------------------------------------
    // Accumulates values from low to high with relaxed memory order
    static void AtomicAccumulatorRelaxed(unsigned long low, unsigned long high)
    {
        for (unsigned long i = low; i < high; i++)
        {
            benchmark::DoNotOptimize(_atomic_sum.fetch_add(i, std::memory_order_relaxed));
        }
    }

    //-----------------------------------------------------
    // Accumulates values from low to high using reduction and relaxed memory
    // order
    static void AtomicAccumulatorPartition(unsigned long low, unsigned long high)
    {
        unsigned long local_sum = 0;
        for (unsigned long i = low; i < high; i++)
        {
            benchmark::DoNotOptimize(local_sum += i);
        }
        _atomic_sum.fetch_add(local_sum, std::memory_order_seq_cst);
    }

    //-----------------------------------------------------
    // Accumulates values from low to high using reduction and relaxed memory
    // order
    static void AtomicAccumulatorPartitionRelaxed(unsigned long low, unsigned long high)
    {
        unsigned long local_sum = 0;
        for (unsigned long i = low; i < high; i++)
        {
            benchmark::DoNotOptimize(local_sum += i);
        }
        _atomic_sum.fetch_add(local_sum, std::memory_order_relaxed);
    }

    //-----------------------------------------------------
    // Accumulates values from low to high using reduction using mutex
    static void MutexAccumulatorPartition(unsigned long low, unsigned long high)
    {
        unsigned long local_sum = 0;
        for (unsigned long i = low; i < high; i++)
        {
            benchmark::DoNotOptimize(local_sum += i);
        }

        {
            std::lock_guard<std::mutex> lg(_my_mutex);
            _mutex_sum += local_sum;
        }
    }

    //-----------------------------------------------------
    template<class T>
    static void Driver(unsigned long number_of_threads, T func, unsigned long size)
    {
        Init();
        std::vector<std::thread> threads;

        unsigned long step       = size / number_of_threads;
        unsigned long num_chunks = step == 0 ? 1 : std::ceil((double)size / (double)step);

        for (unsigned long i = 0; i < num_chunks; i++)
        {
            unsigned long low  = i * step;
            unsigned long high = std::min((i + 1) * step, (unsigned long)size);
            high               = std::max((unsigned long)0, high);
            threads.push_back(std::thread([&func, low, high]() { func(low, high); }));
        }

        for (std::thread &t : threads)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }

    //-----------------------------------------------------
public:
    static std::mutex                 _my_mutex;
    static unsigned long              _mutex_sum;
    static std::atomic<unsigned long> _atomic_sum;
};

#endif /* __RANGE_ACCUMULATOR_HPP__ */