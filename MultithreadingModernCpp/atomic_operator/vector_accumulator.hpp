#ifndef __VECTOR_ACCUMULATOR_HPP__
#define __VECTOR_ACCUMULATOR_HPP__

#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// A class for comparison of lockless and lockfull multithreading
class VectorAccumulator
{
public:
    static void Init()
    {
        _mutex_sum  = 0;
        _atomic_sum = 0;
    }

    static void AtomicAccumulator(std::vector<unsigned long> &a, unsigned long low, unsigned long high)
    {
        for (unsigned long i = low; i < high; i++)
        {
            _atomic_sum += a[i];
        }
    }

    //-----------------------------------------------------
    static void AtomicAccumulatorRelaxed(std::vector<unsigned long> &a, unsigned long low, unsigned long high)
    {
        for (unsigned long i = low; i < high; i++)
        {
            _atomic_sum.fetch_add(a[i], std::memory_order_relaxed);
        }
    }

    //-----------------------------------------------------
    static void AtomicAccumulatorPartitionRelaxed(std::vector<unsigned long> &a, unsigned long low, unsigned long high)
    {
        unsigned long local_sum = 0;
        for (unsigned long i = low; i < high; i++)
        {
            local_sum += a[i];
        }
        _atomic_sum.fetch_add(local_sum, std::memory_order_relaxed);
    }

    //-----------------------------------------------------

    static void MutexAccumulatorPartition(std::vector<unsigned long> &a, unsigned long low, unsigned long high)
    {
        unsigned long local_sum = 0;
        for (unsigned long i = low; i < high; i++)
        {
            local_sum += a[i];
        }

        {
            std::lock_guard<std::mutex> lg(_my_mutex);
            _mutex_sum += local_sum;
        }
    }

    //-----------------------------------------------------
    template<class T>
    static void Driver(std::vector<unsigned long> &a, unsigned long number_of_threads, T func)
    {
        Init();
        std::vector<std::thread> threads;

        unsigned long step       = a.size() / number_of_threads;
        unsigned long num_chunks = step == 0 ? 1 : std::ceil((double)a.size() / (double)step);

        for (unsigned long i = 0; i < num_chunks; i++)
        {
            unsigned long low  = i * step;
            unsigned long high = std::min((i + 1) * step, (unsigned long)a.size());
            high               = std::max((unsigned long)0, high);
            threads.push_back(std::thread([&a, &func, low, high]() { func(a, low, high); }));
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

#endif /* __VECTOR_ACCUMULATOR_HPP__ */