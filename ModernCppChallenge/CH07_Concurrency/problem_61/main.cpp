/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Parallel transform algorithm
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 * In C++17, a series of standard general-purpose algorithms, 
 * including std::transform(), have overloads that implement 
 * a parallel version of the algorithm that can be executed according to
 * a specified execution policy.
 * 
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <utility>
#include <vector>

/**
 * @brief Parallel transform algorithm
 * 
 * Write a general-purpose algorithm that applies a given unary function
 * to transform the elements of a range in parallel. The unary operation used 
 * to transform the range must not invalidate range iterators or modify the elements 
 * of the range. The level of parallelism, that is, the number of execution threads 
 * and the way it is achieved, is an implementation detail.
 * 
 * The general-purpose function std::transform() applies a given function to a range
 *  and stores the result in another (or the same) range. The requirement for this problem
 *  is implementing a parallel version of such a function. A general-purpose one would
 *  take iterators as arguments to define the first and one-past-last element of the range.
 * Because the unary function is applied in the same manner to all the elements of
 *  the range, it is fairly simple to parallelize the operation. For this task, 
 * we will be using threads. Since it is not specified how many threads should be running
 *  at the same time, we could use std::thread::hardware_concurrency(). 
 * This function returns a hint for the number of concurrent threads
 * supported by the implementation.
 * 
 * A parallel version of the algorithm performs better than a sequential implementation
 *  only if the size of the range exceeds a particular threshold, which may vary with
 *  compilation options, platform, or hardware. In the following implementation that
 *  threshold is set to 10,000 elements. As a further exercise, you could experiment 
 * with various thresholds and range sizes to see how the execution time changes.
 * 
 * The following function, ptransform(), implements the parallel transform algorithm 
 * as requested. It simply calls std::transform() if the range size does not exceed 
 * the defined threshold. Otherwise, it splits the range into several equal parts, 
 * one for each thread, and calls std::transform() on each thread for a particular
 *  subrange. In this case, the function blocks the calling thread until 
 * all the worker threads finish execution:
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename T, typename Func>
std::vector<T> alter(std::vector<T> data, Func &&func)
{
    std::transform(std::begin(data), std::end(data), std::begin(data), std::forward<Func>(func));

    return data;
}

template<typename T, typename Func>
std::vector<T> palter(std::vector<T> data, Func &&func)
{
    if (data.size() <= 10000)
    {
        std::transform(std::begin(data), std::end(data), std::begin(data), std::forward<Func>(func));
    }
    else
    {
        std::vector<std::thread> threads;

        int thread_count = std::thread::hardware_concurrency();

        auto first = std::begin(data);
        auto last  = first;
        auto size  = data.size() / thread_count;

        for (size_t i = 0; i < thread_count; ++i)
        {
            first = last;
            last  = i == thread_count - 1 ? std::end(data) : first + size;

            threads.emplace_back([first, last, &func]()
                                 { std::transform(first, last, first, std::forward<Func>(func)); });
        }

        for (size_t i = 0; i < thread_count; ++i)
        {
            threads[i].join();
        }
    }

    return data;
}

template<typename RandomAccessIterator, typename Func>
void ptransform(RandomAccessIterator begin, RandomAccessIterator end, Func &&func)
{
    auto size = std::distance(begin, end);
    if (size <= 10000)
    {
        std::transform(begin, end, begin, std::forward<Func>(func));
    }
    else
    {
        std::vector<std::thread> threads;

        int thread_count = 10;

        auto first = begin;
        auto last  = first;
        size /= thread_count;

        for (size_t i = 0; i < thread_count; ++i)
        {
            first = last;
            if (i == thread_count - 1)
            {
                last = end;
            }
            else
            {
                std::advance(last, size);
            }

            threads.emplace_back([first, last, &func]()
                                 { std::transform(first, last, first, std::forward<Func>(func)); });
        }
        for (auto &t : threads)
        {
            t.join();
        }
    }
}

template<typename T, typename Func>
std::vector<T> palter2(std::vector<T> data, Func &&func)
{
    ptransform(std::begin(data), std::end(data), std::forward<Func>(func));

    return data;
}

// ------------------------------
int main(int argc, char **argv)
{
    const size_t     count = 10000000;
    std::vector<int> data(count);

    std::random_device rd;
    std::mt19937       mt;

    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    mt.seed(seq);
    std::uniform_int_distribution<> ud(1, 100);

    std::generate_n(std::begin(data), count, [&mt, &ud]() { return ud(mt); });

    auto start = std::chrono::system_clock::now();
    auto r1    = alter(data, [](const int e) { return e * e; });
    auto end   = std::chrono::system_clock::now();
    auto t1    = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "time: " << t1.count() << "ms" << std::endl;

    start   = std::chrono::system_clock::now();
    auto r2 = palter(data, [](const int e) { return e * e; });
    end     = std::chrono::system_clock::now();
    auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "time: " << t2.count() << "ms" << std::endl;

    start   = std::chrono::system_clock::now();
    auto r3 = palter2(data, [](const int e) { return e * e; });
    end     = std::chrono::system_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "time: " << t3.count() << "ms" << std::endl;

    // 断言不同计算方式的结果是一致的
    assert(r1 == r2);
    assert(r1 == r3);

    return 0;
}
