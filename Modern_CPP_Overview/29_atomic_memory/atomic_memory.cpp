/**
 * @file atomic_memory.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Atomic Operation and Memory Model
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> count = {0};

// 内存对齐
struct A
{
    float x;
    int y;
    long long z;
};

int main(int argc, char const *argv[])
{
    std::thread thread_1([]()
                         { count.fetch_add(1); });

    std::thread thread_2([]()
                         {
                             count++;    /* identical to fetch_add */
                             count += 1; /* identical to fetch_add */
                         });

    thread_1.join();
    thread_2.join();
    std::cout << count << std::endl;

    /* ----------------------------- */
    // std::atomic<A> a;
    // std::cout << std::boolalpha << a.is_lock_free() << std::endl;

    return 0;
}
