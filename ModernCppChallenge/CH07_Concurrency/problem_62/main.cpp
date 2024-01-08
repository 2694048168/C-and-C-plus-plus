/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Parallel min and max element algorithms using threads
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 * You can take it as a further exercise to implement yet another general-purpose
 *  algorithm that computes the sum of all the elements of a range in parallel using threads.
 * 
 */

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

/**
 * @brief Parallel min and max element algorithms using threads
 * 
 * Implement general-purpose parallel algorithms that find the minimum value
 * and, respectively, the maximum value in a given range. The parallelism should be
 * implemented using threads, although the number of concurrent threads is an implementation detail.
 * 
 */

template<typename Iterator, typename Func>
auto sprocess(Iterator begin, Iterator end, Func &&func)
{
    return std::forward<Func>(func)(begin, end);
}

template<typename Iterator>
auto smin(Iterator begin, Iterator end)
{
    return sprocess(begin, end, [](auto b, auto e) { return *std::min_element(b, e); });
}

template<typename Iterator>
auto smax(Iterator begin, Iterator end)
{
    return sprocess(begin, end, [](auto b, auto e) { return *std::max_element(b, e); });
}

/**
 * @brief Solution:
 * What is slightly different is that the function concurrently executing 
 * on each thread must return a value that represents the minimum or the maximum element
 *  in the subrange. The pprocess() function template, shown as follows, 
 * is a higher-level function that implements the requested functionality generically,
 *  in the following way:
 * 1. Its arguments are the first and one-past-last iterators to the range
 *    and a function object that processes the range that we will call f.
 * 2. If the size of the range is smaller than a particular threshold, set to 10,000
 *    elements here, it simply executes the function object f received as argument.
 * 3. Otherwise, it splits the input range into a number of subranges of equal size,
 *    one for each concurrent thread that could be executed. 
 *    Each thread runs f for the selected subrange.
 * 4. The results of the parallel execution of f are collected in an std::vector, 
 *    and after the execution of all threads is completed, f is used again to determine
 *    the overall result from the intermediate results:
------------------------------------------------------ */

template<typename Iterator, typename Func>
auto pprocess(Iterator begin, Iterator end, Func &&func)
{
    auto size = std::distance(begin, end);
    if (size <= 10000)
    {
        return std::forward<Func>(func)(begin, end);
    }
    else
    {
        int thread_count = std::thread::hardware_concurrency();

        std::vector<std::thread>                                         threads;
        std::vector<typename std::iterator_traits<Iterator>::value_type> mins(thread_count);

        auto first = begin;
        auto last  = first;
        size /= thread_count;
        for (size_t i = 0; i < thread_count; ++i)
        {
            first = last;
            if (i == thread_count - 1)
                last = end;
            else
                std::advance(last, size);

            threads.emplace_back([first, last, &func, &r = mins[i]]() { r = std::forward<Func>(func)(first, last); });
        }

        for (auto &t : threads)
        {
            t.join();
        }

        return std::forward<Func>(func)(std::begin(mins), std::end(mins));
    }
}

/**
 * @brief Two functions, called pmin() and pmax(),
 *  are provided to implement the required general-purpose min and max parallel algorithms.
 *  These two are in turn calling pprocess(), passing for the third argument a lambda that
 *  uses either the std::min_element() or the std::max_element() standard algorithm:
 ---------------------------------------------------------------------------------- */
template<typename Iterator>
auto pmin(Iterator begin, Iterator end)
{
    return pprocess(begin, end, [](auto b, auto e) { return *std::min_element(b, e); });
}

template<typename Iterator>
auto pmax(Iterator begin, Iterator end)
{
    return pprocess(begin, end, [](auto b, auto e) { return *std::max_element(b, e); });
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
    std::uniform_int_distribution<> ud(1, 1000);

    std::generate_n(std::begin(data), count, [&mt, &ud]() { return ud(mt); });

    {
        std::cout << "======== minimum element ========" << std::endl;

        auto start = std::chrono::system_clock::now();
        auto r1    = smin(std::begin(data), std::end(data));
        auto end   = std::chrono::system_clock::now();
        auto t1    = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "seq time: " << t1.count() << "ms" << std::endl;

        start   = std::chrono::system_clock::now();
        auto r2 = pmin(std::begin(data), std::end(data));
        end     = std::chrono::system_clock::now();
        auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "par time: " << t2.count() << "ms" << std::endl;

        assert(r1 == r2);
    }

    {
        std::cout << "======== maximum element ========" << std::endl;

        auto start = std::chrono::system_clock::now();
        auto r1    = smax(std::begin(data), std::end(data));
        auto end   = std::chrono::system_clock::now();
        auto t1    = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "seq time: " << t1.count() << "ms" << std::endl;

        start   = std::chrono::system_clock::now();
        auto r2 = pmax(std::begin(data), std::end(data));
        end     = std::chrono::system_clock::now();
        auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "par time: " << t2.count() << "ms" << std::endl;

        assert(r1 == r2);
    }

    return 0;
}
