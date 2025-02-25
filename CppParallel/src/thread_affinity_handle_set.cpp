/**
 * @file thread_affinity_handle_set.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief This program shows off setting thread affinity
 * @version 0.1
 * @date 2025-02-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "benchmark/benchmark.h"

#ifdef _WIN32
#    include <Windows.h>
#else
#    include <pthread.h>
#endif

#include <atomic>
#include <cassert>
#include <thread>

// -------------------------------------
// Aligned type to avoid atomics on the same cache line
struct AlignedAtomic
{
    alignas(64) std::atomic<int> val = 0;
};

void thread_affinity()
{
    AlignedAtomic a;
    AlignedAtomic b;

    // Work function for our threads
    auto work = [](AlignedAtomic &atomic)
    {
        for (int i = 0; i < (1 << 20); i++) atomic.val++;
    };

    // Create thread 0 and 1, and pin them to core 0
    std::thread t0(work, std::ref(a));
    std::thread t1(work, std::ref(a));
    // Create thread 1 and 2, and pin them to core 1
    std::thread t2(work, std::ref(b));
    std::thread t3(work, std::ref(b));

#ifdef _WIN32
    //设置线程CPU亲和力
    ::SetThreadAffinityMask(t0.native_handle(), 0x0001);
    ::SetThreadAffinityMask(t1.native_handle(), 0x0002);
    ::SetThreadAffinityMask(t2.native_handle(), 0x0004);
    ::SetThreadAffinityMask(t3.native_handle(), 0x0008);
    //设置线程优先级 相对优先级
    // ::SetThreadPriority(t0.native_handle(), 16);
#else
#    // Create cpu sets for threads 0,1 and 2,3
    cpu_set_t cpu_set_0;
    cpu_set_t cpu_set_1;

    // Zero them out
    CPU_ZERO(&cpu_set_0);
    CPU_ZERO(&cpu_set_1);

    // And set the CPU cores we want to pin the threads too
    CPU_SET(0, &cpu_set_0);
    CPU_SET(1, &cpu_set_1);

    assert(pthread_setaffinity_np(t0.native_handle(), sizeof(cpu_set_t), &cpu_set_0) == 0);
    assert(pthread_setaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &cpu_set_0) == 0);

    assert(pthread_setaffinity_np(t2.native_handle(), sizeof(cpu_set_t), &cpu_set_1) == 0);
    assert(pthread_setaffinity_np(t3.native_handle(), sizeof(cpu_set_t), &cpu_set_1) == 0);
#endif

    // Join the threads
    t0.join();
    t1.join();
    t2.join();
    t3.join();
}

// Thread affinity benchmark
static void thread_affinity_(benchmark::State &s)
{
    for (auto _ : s)
    {
        thread_affinity();
    }
}

BENCHMARK(thread_affinity_)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
