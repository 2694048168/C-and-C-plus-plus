/**
 * @file 14_synchronization_strategy.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 调虎离山: 多线程编程中的资源竞争和同步策略
 * @version 0.1
 * @date 2026-04-06
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 14_synchronization_strategy.exe 14_synchronization_strategy.c
 * clang -o 14_synchronization_strategy.exe 14_synchronization_strategy.c
 * 
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

// 高精度计时（微秒）
static double get_time_us(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / freq.QuadPart * 1000000.0;
}

// ========== 无同步版本（错误，竞态条件） ==========
volatile long long counter_no_sync = 0; // volatile 只防止优化，不保证原子性

DWORD WINAPI increment_no_sync(LPVOID arg)
{
    int n = *(int *)arg;
    for (int i = 0; i < n; ++i)
    {
        counter_no_sync++; // 非原子操作，会产生数据竞争
    }
    return 0;
}

// ========== 临界区版本（互斥锁的Windows实现） ==========
CRITICAL_SECTION cs;
long long counter_cs = 0;

DWORD WINAPI increment_cs(LPVOID arg)
{
    int n = *(int *)arg;
    for (int i = 0; i < n; ++i)
    {
        EnterCriticalSection(&cs);
        counter_cs++;
        LeaveCriticalSection(&cs);
    }
    return 0;
}

// ========== 原子操作版本（使用InterlockedIncrement） ==========
volatile LONG counter_interlock = 0; // Interlocked* 要求 LONG 类型

DWORD WINAPI increment_interlock(LPVOID arg)
{
    int n = *(int *)arg;
    for (int i = 0; i < n; ++i)
    {
        InterlockedIncrement(&counter_interlock);
    }
    return 0;
}

// 通用测试函数
void run_test(const char *name, LPTHREAD_START_ROUTINE thread_func, int threads, int increments_per_thread)
{
    HANDLE *handles = malloc(threads * sizeof(HANDLE));
    int *arg = malloc(sizeof(int));
    *arg = increments_per_thread;

    double start = get_time_us();
    for (int i = 0; i < threads; ++i)
    {
        handles[i] = CreateThread(NULL, 0, thread_func, arg, 0, NULL);
        if (!handles[i])
        {
            fprintf(stderr, "CreateThread 失败\n");
            exit(EXIT_FAILURE);
        }
    }
    WaitForMultipleObjects(threads, handles, TRUE, INFINITE);
    double elapsed = (get_time_us() - start) / 1000.0; // ms

    // 关闭线程句柄
    for (int i = 0; i < threads; ++i)
        CloseHandle(handles[i]);

    long long expected = (long long)threads * increments_per_thread;
    long long actual = 0;
    if (thread_func == increment_no_sync)
        actual = counter_no_sync;
    else if (thread_func == increment_cs)
        actual = counter_cs;
    else if (thread_func == increment_interlock)
        actual = counter_interlock;

    printf("%-16s | 预期值: %-8lld | 实际值: %-8lld | 耗时: %.2f ms | 结果: %s\n", name, expected, actual, elapsed,
           (expected == actual) ? "正确" : "错误(竞态)");

    free(handles);
    free(arg);
}

int main(void)
{
    // 设置控制台代码页为UTF-8（可选，防止中文乱码）
    SetConsoleOutputCP(CP_UTF8);

    const int THREADS = 4;
    const int INCREMENTS_PER_THREAD = 100000;

    printf("调虎离山：多线程自增同步策略对比 (Windows API)\n");
    printf("线程数=%d, 每线程自增次数=%d, 预期最终值=%lld\n\n", THREADS, INCREMENTS_PER_THREAD,
           (long long)THREADS * INCREMENTS_PER_THREAD);

    // 初始化临界区
    InitializeCriticalSection(&cs);

    // 重置计数器
    counter_no_sync = 0;
    counter_cs = 0;
    counter_interlock = 0;

    // 测试无同步（危险）
    run_test("无同步(竞态)", increment_no_sync, THREADS, INCREMENTS_PER_THREAD);

    // 重置
    counter_cs = 0;
    run_test("临界区", increment_cs, THREADS, INCREMENTS_PER_THREAD);

    // 重置
    counter_interlock = 0;
    run_test("原子操作", increment_interlock, THREADS, INCREMENTS_PER_THREAD);

    // 清理临界区
    DeleteCriticalSection(&cs);

    printf("\n调虎离山核心思想：\n");
    printf("- 无同步时，线程直接争夺共享变量 → 数据损坏\n");
    printf("- 临界区：将争夺转移到锁上，一次只放一个线程进临界区\n");
    printf("- 原子操作：硬件层面保证操作不可分割，无阻塞\n");

    return 0;
}
