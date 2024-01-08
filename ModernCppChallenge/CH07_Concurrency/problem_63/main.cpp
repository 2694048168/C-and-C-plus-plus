/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Parallel min and max element algorithms using asynchronous functions
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 * You can again take it as a further exercise to implement a general-purpose
 * algorithm that computes the sum of all the elements of a range
 *  in parallel using asynchronous functions.
 * 
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>
#include <utility>
#include <vector>

/**
 * @brief Parallel min and max element algorithms using asynchronous functions
 * 
 * Implement general-purpose parallel algorithms that find the minimum value and,
 * respectively, the maximum value in a given range. The parallelism should be implemented
 * using asynchronous functions, although the number of concurrent functions is an implementation detail.
 *
 * The only difference between this problem and the previous one is how the parallelism
 *  is achieved. For the previous problem, the use of threads was required. For this one,
 *  you must use asynchronous functions. A function can be executed asynchronously 
 * with std::async(). This function creates a promise, which is an asynchronous provider
 * of the result of a function executed asynchronously. A promise has a shared state
 *  (which can store either the return value of a function or an exception that resulted from
 *  the execution of the function) and an associated future object that provides access to
 *  the shared state from a different thread. The promise-future pair defines a channel
 *  that enables communicating values across threads. 
 * std::async() returns the future associated with the promise it creates.
 *
 * In the following implementation of pprocess(), the use of threads from 
 * the previous version has been replaced with calls to std::async(). 
 * Note that you must specify std::launch::async as the first parameter 
 * to std::async() to guarantee an asynchronous execution and not a lazy evaluation.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename Iterator, typename F>
auto sprocess(Iterator begin, Iterator end, F &&f)
{
    return std::forward<F>(f)(begin, end);
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

template<typename Iterator, typename F>
auto pprocess(Iterator begin, Iterator end, F &&f)
{
    auto size = std::distance(begin, end);
    if (size <= 10000)
    {
        return std::forward<F>(f)(begin, end);
    }
    else
    {
        int task_count = std::thread::hardware_concurrency();

        std::vector<std::future<typename std::iterator_traits<Iterator>::value_type>> tasks;

        auto first = begin;
        auto last  = first;
        size /= task_count;
        for (int i = 0; i < task_count; ++i)
        {
            first = last;
            if (i == task_count - 1)
                last = end;
            else
                std::advance(last, size);

            tasks.emplace_back(
                std::async(std::launch::async, [first, last, &f]() { return std::forward<F>(f)(first, last); }));
        }

        std::vector<typename std::iterator_traits<Iterator>::value_type> mins;
        for (auto &t : tasks) mins.push_back(t.get());

        return std::forward<F>(f)(std::begin(mins), std::end(mins));
    }
}

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
    auto               seed_data = std::array<int, std::mt19937::state_size>{};
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
