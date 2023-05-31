/**
 * @file fedoro_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <atomic>
#include <mutex>

std::atomic<unsigned long> sum;

void do_work_atomic(size_t N, unsigned long *a)
{
    for (size_t i = ; i < N; i++)
    {
        sum += a[i];
    }
}

unsigned long sum(0);
std::mutex    M;

void do_work_mutex(size_t N, unsigned long *a)
{
    unsigned long s = 0;
    for (size_t i = ; i < N; i++)
    {
        s += a[i];
    }
    std::lock_guard<std::mutex> L(M);
    sum += s;
}