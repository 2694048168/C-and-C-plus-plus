/**
 * @file memory_order.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Memory Orders
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <atomic>
#include <thread>
#include <vector>
#include <iostream>

const int N = 10000;

void relaxed_order()
{
    std::cout << "relaxed_order: " << std::endl;

    std::atomic<int> counter = {0};
    std::vector<std::thread> vt;
    for (int i = 0; i < N; ++i)
    {
        vt.emplace_back([&]()
                        { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto &t : vt)
    {
        t.join();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = (t2 - t1).count();
    std::cout << "relaxed order speed: " << duration / N << "ns" << std::endl;
}

void release_consume_order()
{
    std::cout << "release_consume_order: " << std::endl;

    std::atomic<int *> ptr;
    int v;
    std::thread producer([&]()
                         {
                             int *p = new int(42);
                             v = 1024;
                             ptr.store(p, std::memory_order_release);
                         });
    std::thread consumer([&]()
                         {
                             int *p;
                             while (!(p = ptr.load(std::memory_order_consume)))
                                 ;

                             std::cout << "p: " << *p << std::endl;
                             std::cout << "v: " << v << std::endl;
                         });
    producer.join();
    consumer.join();
}

void release_acquire_order()
{
    std::cout << "release_acquire_order: " << std::endl;

    int v;
    std::atomic<int> flag = {0};
    std::thread release([&]()
                        {
                            v = 42;
                            flag.store(1, std::memory_order_release);
                        });
    std::thread acqrel([&]()
                       {
                           int expected = 1; // must before compare_exchange_strong
                           while (!flag.compare_exchange_strong(expected, 2, std::memory_order_acq_rel))
                           {
                               expected = 1; // must after compare_exchange_strong
                           }
                           // flag has changed to 2
                       });
    std::thread acquire([&]()
                        {
                            while (flag.load(std::memory_order_acquire) < 2)
                                ;

                            std::cout << "v: " << v << std::endl; // must be 42
                        });
    release.join();
    acqrel.join();
    acquire.join();
}

void sequential_consistent_order()
{
    std::cout << "sequential_consistent_order: " << std::endl;

    std::atomic<int> counter = {0};
    std::vector<std::thread> vt;
    for (int i = 0; i < N; ++i)
    {
        vt.emplace_back([&]()
                        { counter.fetch_add(1, std::memory_order_seq_cst); });
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto &t : vt)
    {
        t.join();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = (t2 - t1).count();
    std::cout << "sequential consistent speed: " << duration / N << "ns" << std::endl;
}

int main(int argc, char const *argv[])
{
    relaxed_order();
    release_consume_order();
    release_acquire_order();
    sequential_consistent_order();
    return 0;
}
