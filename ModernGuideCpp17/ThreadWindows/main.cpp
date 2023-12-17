/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Windows API 多线程
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <windows.h>

#include <iostream>

/**
 * @brief 生产者消费者问题是一个著名的线程同步问题
 * 有一个生产者在生产产品，这些产品将提供给若干个消费者去消费，为了使生产者和消费者能并发执行，
 * 在两者之间设置一个具有多个缓冲区的缓冲池，
 * 生产者将它生产的产品放入一个缓冲区中，消费者可以从缓冲区中取走产品进行消费，
 * 显然生产者和消费者之间必须保持同步，
 * 即不允许消费者到一个空的缓冲区中取产品，也不允许生产者向一个已经放入产品的缓冲区中再次投放产品。
 *
 * (一) 一个生产者，一个消费者，一个缓冲区。
 * 要满足生产者与消费者关系，我们需要保证以下两点：
 * 第一．从缓冲区取出产品和向缓冲区投放产品必须是互斥进行的。可以用关键段和互斥量来完成。
 * 第二．生产者要等待缓冲区为空，这样才可以投放产品，消费者要等待缓冲区不为空，
 *  这样才可以取出产品进行消费。并且由于有二个等待过程，所以要用二个事件或信号量来控制。
 *
 * (二) 一个生产者，两个消费者，一个缓冲池（四个缓冲区）
 * 相比于一个生产者，一个消费者，一个缓冲区，生产者由一个变成多个不难处理，多开线程就可以，
 * 需要注意的是缓冲区的变化，可以利用两个信号量就可以解决这种缓冲池有多个缓冲区的情况。
 * 用一个信号量A来记录为空的缓冲区个数，另一个信号量B记录非空的缓冲区个数，
 * 然后生产者等待信号量A，消费者等待信号量B就可以了。
 * 
 */

//生产者消费者问题,一个生产者，一个消费者，一个缓冲区。
DWORD WINAPI ProducerThread(LPVOID);
DWORD WINAPI ConsumerThread(LPVOID);

const int        PRODUCT_NUM = 10; //总共生产10个产品
int              g_Buffer    = 0;  //缓冲区
CRITICAL_SECTION g_csVar;          //互斥锁
HANDLE           g_hEventBufEmpty, g_hEventBufFull;

// ====================================
int main(int argc, const char **argv)
{
    // 生产者消费者问题,一个生产者，一个消费者，一个缓冲区
    InitializeCriticalSection(&g_csVar);
    g_hEventBufEmpty = CreateEvent(NULL, false, true, NULL);  //缓冲区为空事件
    g_hEventBufFull  = CreateEvent(NULL, false, false, NULL); //缓冲区满事件

    const int THREAD_NUM = 2;
    HANDLE    handle[THREAD_NUM];
    handle[0] = CreateThread(NULL, 0, ProducerThread, NULL, 0, NULL); //生产者线程
    handle[1] = CreateThread(NULL, 0, ConsumerThread, NULL, 0, NULL); //消费者线程
    WaitForMultipleObjects(THREAD_NUM, handle, true, INFINITE);

    DeleteCriticalSection(&g_csVar);
    CloseHandle(handle[0]);
    CloseHandle(handle[1]);
    CloseHandle(g_hEventBufEmpty);
    CloseHandle(g_hEventBufFull);

    return 0;
}

DWORD WINAPI ProducerThread(LPVOID p)
{
    for (int i = 1; i <= PRODUCT_NUM; i++)
    {
        WaitForSingleObject(g_hEventBufEmpty, INFINITE); //等待缓冲区为空
        EnterCriticalSection(&g_csVar);
        g_Buffer = i;
        std::cout << "生产者将数据 " << g_Buffer << " 放入缓冲区！" << std::endl;
        LeaveCriticalSection(&g_csVar);
        SetEvent(g_hEventBufFull); //触发事件，缓冲区满
    }

    return 0;
}

DWORD WINAPI ConsumerThread(LPVOID p)
{
    for (int i = 1; i <= PRODUCT_NUM; i++)
    {
        WaitForSingleObject(g_hEventBufFull, INFINITE); //等待缓冲区满
        EnterCriticalSection(&g_csVar);
        std::cout << "\t\t\t\t消费者将数据 " << g_Buffer << " 从缓冲区取出！" << std::endl;
        LeaveCriticalSection(&g_csVar);
        SetEvent(g_hEventBufEmpty); //触发事件，清空缓冲区
    }
    return 0;
}
