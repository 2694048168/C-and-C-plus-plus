/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Parallel sort algorithm
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

#include <array>
#include <cassert>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief Parallel sort algorithm
 * 
 * Write a parallel version of the sort algorithm as defined for problem 53. 
 * Sort Algorithm, in Chapter 6, Algorithms and Data Structures, 
 * which, given a pair of random access iterators to define its lower and upper bounds,
 * sorts the elements of the range using the quicksort algorithm. 
 * The function should use the comparison operators for comparing the elements of the range.
 * The level of parallelism and the way to achieve it is an implementation detail.
 * 
 */

/**
 * @brief Solution:
 *
 * We saw a sequential implementation of the quicksort algorithm earlier. 
 * Quicksort is a divide and conquer algorithm that relies on partitioning the range
 *  to be sorted into two parts, one that contains only elements smaller than a selected
 *  element, called the pivot, and one that contains only elements greater than the pivot.
 * It then proceeds to recursively apply the same algorithm on the two partitions, 
 * until the partitions have only one element or none. Because of the nature of 
 * the algorithm, quicksort can be easily parallelized to recursively apply the algorithm on the two partitions concurrently.
 *
 * The pquicksort() function uses asynchronous functions for this purpose. 
 * However, parallelization is only efficient for larger ranges. 
 * There is a threshold under which the overhead with context switches for 
 * parallel execution is too large and the parallel execution time is greater 
 * than the sequential execution time. In the following implementation, 
 * this threshold is set to 100,000 elements, but as a further exercise, 
 * you could experiment with setting different values and see how the parallel version
 *  performs compared to the sequential one:
------------------------------------------------------ */

template<class RandomIt>
RandomIt partition(RandomIt first, RandomIt last)
{
    auto pivot = *first;
    auto i     = first + 1;
    auto j     = last - 1;
    while (i <= j)
    {
        while (i <= j && *i <= pivot) i++;
        while (i <= j && *j > pivot) j--;
        if (i < j)
            std::iter_swap(i, j);
    }

    std::iter_swap(i - 1, first);

    return i - 1;
}

template<class RandomIt, class Compare>
RandomIt partitionc(RandomIt first, RandomIt last, Compare comp)
{
    auto pivot = *first;
    auto i     = first + 1;
    auto j     = last - 1;
    while (i <= j)
    {
        while (i <= j && comp(*i, pivot)) i++;
        while (i <= j && !comp(*j, pivot)) j--;
        if (i < j)
            std::iter_swap(i, j);
    }

    std::iter_swap(i - 1, first);

    return i - 1;
}

template<class RandomIt>
void quicksort(RandomIt first, RandomIt last)
{
    if (first < last)
    {
        auto p = partition(first, last);
        quicksort(first, p);
        quicksort(p + 1, last);
    }
}

template<class RandomIt, class Compare>
void quicksort(RandomIt first, RandomIt last, Compare comp)
{
    if (first < last)
    {
        auto p = partitionc(first, last, comp);
        quicksort(first, p, comp);
        quicksort(p + 1, last, comp);
    }
}

template<class RandomIt>
void pquicksort(RandomIt first, RandomIt last)
{
    if (first < last)
    {
        auto p = partition(first, last);

        if (last - first <= 100000)
        {
            pquicksort(first, p);
            pquicksort(p + 1, last);
        }
        else
        {
            auto f1 = std::async(std::launch::async, [first, p]() { pquicksort(first, p); });
            auto f2 = std::async(std::launch::async, [last, p]() { pquicksort(p + 1, last); });
            f1.wait();
            f2.wait();
        }
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    const size_t     count = 1000000;
    std::vector<int> data(count);

    std::random_device rd;
    std::mt19937       mt;
    auto               seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    mt.seed(seq);
    std::uniform_int_distribution<> ud(1, 1000);

    std::cout << "======== generating..." << std::endl;
    std::generate_n(std::begin(data), count, [&mt, &ud]() { return ud(mt); });

    auto d1 = data;
    auto d2 = data;

    std::cout << "======== sorting..." << std::endl;
    auto start = std::chrono::system_clock::now();
    quicksort(std::begin(d1), std::end(d1));
    auto end = std::chrono::system_clock::now();
    auto t1  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "======== quicksort time: " << t1.count() << "ms" << std::endl;

    start = std::chrono::system_clock::now();
    pquicksort(std::begin(d2), std::end(d2));
    end     = std::chrono::system_clock::now();
    auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "======== pquicksort time: " << t2.count() << "ms" << std::endl;

    std::cout << "======== verifying..." << std::endl;
    assert(d1 == d2);

    return 0;
}

// 有点令人诧异的结果, 需要仔细思考 why
// ======== generating...
// ======== sorting...
// ======== quicksort time: 37863ms
// ======== pquicksort time: 423007ms
// ======== verifying...
