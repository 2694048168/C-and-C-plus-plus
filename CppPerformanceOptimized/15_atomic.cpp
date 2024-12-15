/**
 * @file 15_atomic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Stopwatch.hpp"

#if _WIN32
#    include <windows.h>
#endif

int main(int argc, const char *argv[])
{
#if _WIN32
    SetConsoleOutputCP(65001);
#endif

    constexpr unsigned multiplier = 10;

    {
        Stopwatch{"原子性存储操作测试"};

        typedef unsigned long long counter_t;
        std::atomic<counter_t>     x;
        for (counter_t i = 0, iterations = 10'000'000 * multiplier; i < iterations; ++i)
        {
            x = i;
        }
    }

    {
        Stopwatch{"非原子性存储操作的测试"};

        typedef unsigned long long counter_t;
        counter_t                  x;
        for (counter_t i = 0, iterations = 10'000'000 * multiplier; i < iterations; ++i)
        {
            x = i;
        }
    }

    return 0;
}
