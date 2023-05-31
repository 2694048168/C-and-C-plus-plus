/**
 * @file memory_order_main.cpp
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
#include <numeric>
#include <thread>
#include <vector>

// Demos relaxed memory ordering.
// This will never produce r1 == r2 == 42
int main(int argc, const char **argv)
{
    std::vector<std::thread> threads;

    std::atomic<int> r1, r2, x, y;
    r1 = 0;
    r2 = 0;
    x  = 0;
    y  = 0;

    std::cout << "r1: " << r1 << ", r2: " << r2 << std::endl;
    std::cout << "x: " << x << ", y: " << y << std::endl;

    threads.push_back(std::thread(
        [&x, &r1, &y]
        {
            r1 = x.load(std::memory_order_relaxed);
            if (r1 == 42)
            {
                y.store(r1, std::memory_order_relaxed);
            }
        }));

    threads.push_back(std::thread(
        [&x, &r2, &y]
        {
            r2 = y.load(std::memory_order_relaxed);
            if (r2 == 42)
            {
                x.store(42, std::memory_order_relaxed);
            }
        }));

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "r1: " << r1 << ", r2: " << r2 << std::endl;
    std::cout << "x: " << x << ", y: " << y << std::endl;
    
    return 0;
}
