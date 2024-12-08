/**
 * @file 13_vector_cache_missing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-08
 * 
 * @copyright Copyright (c) 2024
 * 
 * perf tool or valgrind tool
 * command:
valgrind --tool=callgrind --cache-sim=yes --dump-instr=yes --branch-sim=yes
         调用工具          缓存模拟          基于指令集事件计数  分支预测模型
 * 
 */

#include "Stopwatch.hpp"

#include <random>
#include <vector>

template<typename T>
void ordered_insert(T &container, int value)
{
    auto it = container.begin();
    while (it != container.end() && *it < value)
    {
        ++it;
    }
    container.insert(it, value);
}

// -------------------------------------
int main(int argc, const char *argv[])
{
    std::random_device rand;
    std::mt19937       generator(rand());

    std::uniform_int_distribution<> dist;
    std::vector<long long>          list_;

    {
        Stopwatch("========std::list Cache Missing");

        for (int idx{0}; idx < 10000; ++idx)
        {
            ordered_insert(list_, dist(generator));
        }
    }

    return 0;
}
