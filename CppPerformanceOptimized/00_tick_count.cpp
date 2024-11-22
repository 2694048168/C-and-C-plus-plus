/**
 * @file 00_tick_count.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <iostream>
#include <thread>

#ifdef _WIN32
#    include <Windows.h>
#endif

void test_func()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// -----------------------------------
int main(int argc, const char **argv)
{
    DWORD start = GetTickCount();
    for (unsigned i = 0; i < 100; ++i)
    {
        test_func();
    }
    DWORD end = GetTickCount();

    std::cout << "100 calls to Foo() took " << end - start << "ms" << std::endl;
    //! 100 calls to Foo() took 1547ms
    // why is this result? 1547 ms / 100 = 15.47 ms
    // https://learn.microsoft.com/zh-cn/windows/win32/api/sysinfoapi/nf-sysinfoapi-gettickcount
    // GetTickCount 函数的分辨率限制为系统计时器的分辨率，通常范围为 10 毫秒到 16 毫秒
    // GetTickCount 函数的解析不受 GetSystemTimeAdjustment 函数所做的调整的影响
    // 因此测量的基础精度是 15 毫秒，额外的分辨率毫无价值

    // 测量 GetTickCount() 的时标
    unsigned int nz_count = 0;
    unsigned int nz_sum   = 0;
    ULONG        last;
    ULONG        next;
    for (last = GetTickCount(); nz_count < 100; last = next)
    {
        next = GetTickCount();
        if (next != last)
        {
            nz_count += 1;
            nz_sum += (next - last);
        }
    }
    std::cout << "GetTickCount() mean resolution " << (double)nz_sum / nz_count << " ticks" << std::endl;

    // Windows 计时函数的延迟
    ULONG         start_ = GetTickCount();
    LARGE_INTEGER count;
    for (size_t i = 0; i < 100,000,000,000; ++i)
    {
        QueryPerformanceCounter(&count);
    }
    ULONG stop_ = GetTickCount();
    std::cout << stop_ - start_ << " ms for 100m QueryPerformanceCounter() calls" << std::endl;

    return 0;
}
