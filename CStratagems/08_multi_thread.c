/**
 * @file 08_multi_thread.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 隔岸观火: 多线程同步与资源竞争
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o multi_thread.exe 08_multi_thread.c
 * clang -o multi_thread.exe 08_multi_thread.c
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define BUFFER_SIZE 5
#define PRODUCER_COUNT 2
#define CONSUMER_COUNT 3
#define TOTAL_ITEMS 20

int buffer[BUFFER_SIZE];
int in = 0, out = 0;
int produced_count = 0, consumed_count = 0;

CRITICAL_SECTION mutex;
CONDITION_VARIABLE not_full;  // 缓冲区不满条件
CONDITION_VARIABLE not_empty; // 缓冲区不空条件

DWORD WINAPI producer(LPVOID arg)
{
    int id = (int)(intptr_t)arg;

    while (1)
    {
        EnterCriticalSection(&mutex);
        // 当缓冲区满 或 已达生产总量时等待
        while (((in + 1) % BUFFER_SIZE == out) || produced_count >= TOTAL_ITEMS)
        {
            if (produced_count >= TOTAL_ITEMS)
            {
                LeaveCriticalSection(&mutex);
                return 0;
            }
            // 等待"不满"条件被唤醒
            SleepConditionVariableCS(&not_full, &mutex, INFINITE);
        }

        if (produced_count >= TOTAL_ITEMS)
        {
            LeaveCriticalSection(&mutex);
            return 0;
        }

        int item = produced_count + 1;
        buffer[in] = item;
        printf("[生产者 %d] 生产了 %d，位置 %d\n", id, item, in);
        in = (in + 1) % BUFFER_SIZE;
        produced_count++;

        // 唤醒一个等待"不空"的消费者
        WakeConditionVariable(&not_empty);
        LeaveCriticalSection(&mutex);

        Sleep(rand() % 200);
    }
    return 0;
}

DWORD WINAPI consumer(LPVOID arg)
{
    int id = (int)(intptr_t)arg;

    while (1)
    {
        EnterCriticalSection(&mutex);
        // 当缓冲区空 或 已消费完所有物品时等待
        while ((in == out) || consumed_count >= TOTAL_ITEMS)
        {
            if (consumed_count >= TOTAL_ITEMS)
            {
                LeaveCriticalSection(&mutex);
                return 0;
            }
            // 等待"不空"条件被唤醒
            SleepConditionVariableCS(&not_empty, &mutex, INFINITE);
        }

        if (consumed_count >= TOTAL_ITEMS)
        {
            LeaveCriticalSection(&mutex);
            return 0;
        }

        int item = buffer[out];
        printf("[消费者 %d] 消费了 %d，位置 %d\n", id, item, out);
        out = (out + 1) % BUFFER_SIZE;
        consumed_count++;

        // 唤醒一个等待"不满"的生产者
        WakeConditionVariable(&not_full);
        LeaveCriticalSection(&mutex);

        Sleep(rand() % 300);
    }
    return 0;
}

int main(int argc, char const *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    srand(GetCurrentThreadId());

    // 初始化同步对象
    InitializeCriticalSection(&mutex);
    InitializeConditionVariable(&not_full);
    InitializeConditionVariable(&not_empty);

    HANDLE producers[PRODUCER_COUNT];
    HANDLE consumers[CONSUMER_COUNT];

    // 创建生产者线程
    for (int i = 0; i < PRODUCER_COUNT; i++)
    {
        producers[i] = CreateThread(NULL, 0, producer, (LPVOID)(intptr_t)(i + 1), 0, NULL);
    }

    // 创建消费者线程
    for (int i = 0; i < CONSUMER_COUNT; i++)
    {
        consumers[i] = CreateThread(NULL, 0, consumer, (LPVOID)(intptr_t)(i + 1), 0, NULL);
    }

    // 等待所有线程结束
    WaitForMultipleObjects(PRODUCER_COUNT, producers, TRUE, INFINITE);
    WaitForMultipleObjects(CONSUMER_COUNT, consumers, TRUE, INFINITE);

    // 关闭线程句柄
    for (int i = 0; i < PRODUCER_COUNT; i++)
        CloseHandle(producers[i]);
    for (int i = 0; i < CONSUMER_COUNT; i++)
        CloseHandle(consumers[i]);

    // 销毁同步对象
    DeleteCriticalSection(&mutex);

    printf("\n所有生产消费完成！共生产 %d 个，消费 %d 个\n", produced_count, consumed_count);
    return 0;
}
