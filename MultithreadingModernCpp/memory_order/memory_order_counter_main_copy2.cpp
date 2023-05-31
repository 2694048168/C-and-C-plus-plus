/**
 * @file memory_order_counter_main_copy2.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

// Demo Relaxed ordering
std::atomic<int> cnt = {0};

void f()
{
    for (int n = 0; n < 1000; ++n)
    {
        cnt.fetch_add(1, std::memory_order_relaxed);
    }
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<std::thread> v;
    for (int n = 0; n < 10; ++n)
    {
        v.emplace_back(f);
    }
    for (auto &t : v)
    {
        t.join();
    }
    std::cout << "Final counter value is " << cnt << '\n';
}