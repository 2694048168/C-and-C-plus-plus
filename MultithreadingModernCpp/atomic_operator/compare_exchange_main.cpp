/**
 * @file compare_exchange_main.cpp
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
#include <mutex>
#include <thread>
#include <vector>

std::atomic<int> g_atomicX{0};

int f(int a, int b)
{
    return a * a + b;
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
    // compare exchange
    {
        // if (atomic_x == expected) {
        //   atomic_x = desired;
        //   return true;
        // } else {
        //   atomic_x = expected;
        //   return false;
        // }
        std::atomic<int> atomic_x(1);
        int              expected = 2;
        int              desired  = 3;

        bool success = atomic_x.compare_exchange_strong(expected, desired);

        std::cout << "success: " << success << ", atomic_x: " << atomic_x << ", expected: " << expected
                  << ", desired: " << desired << std::endl;

        success = atomic_x.compare_exchange_strong(expected, desired);

        std::cout << "success: " << success << ", atomic_x: " << atomic_x << ", expected: " << expected
                  << ", desired: " << desired << std::endl;
    }

    {
        // Suppose we want to do the following atomically:
        g_atomicX = f(g_atomicX, 10);
        // Or
        g_atomicX = f(g_atomicX.load(), 10);
        // Or
        g_atomicX.store(f(g_atomicX.load(), 10));

        // None of the above alternatives is atomic
        // Using compare_exchange:
        g_atomicX = 4;
        auto oldX = g_atomicX.load();
        while (!g_atomicX.compare_exchange_strong(oldX, f(oldX, 10)))
            ;

        std::cout << "g_atomicX: " << g_atomicX << ", oldX: " << oldX << std::endl;
    }

    return 0;
}