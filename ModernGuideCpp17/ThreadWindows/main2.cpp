/**
 * @file main2.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Windows API 多线程
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <process.h>
#include <stdio.h>
#include <windows.h>
#include <cstring>


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

//1生产者 2消费者 4缓冲区

//设置控制台输出颜色
BOOL SetConsoleColor(WORD wAttributes)
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hConsole == INVALID_HANDLE_VALUE)
        return FALSE;

    return SetConsoleTextAttribute(hConsole, wAttributes);
}

const int        END_PRODUCE_NUMBER = 8; //生产产品个数
const int        BUFFER_SIZE        = 4; //缓冲区个数
int              g_Buffer[BUFFER_SIZE];  //缓冲池
int              g_i, g_j;
//信号量与关键段
CRITICAL_SECTION g_cs;
HANDLE           g_hSemaphoreBufferEmpty, g_hSemaphoreBufferFull;

//生产者线程函数
unsigned int __stdcall ProducerThreadFun(PVOID pM)
{
    for (int i = 1; i <= END_PRODUCE_NUMBER; i++)
    {
        //等待有空的缓冲区出现
        WaitForSingleObject(g_hSemaphoreBufferEmpty, INFINITE);

        //互斥的访问缓冲区
        EnterCriticalSection(&g_cs);
        g_Buffer[g_i] = i;
        printf("生产者在缓冲池第%d个缓冲区中投放数据%d\n", g_i, g_Buffer[g_i]);
        g_i = (g_i + 1) % BUFFER_SIZE;
        LeaveCriticalSection(&g_cs);

        //通知消费者有新数据了
        ReleaseSemaphore(g_hSemaphoreBufferFull, 1, NULL);
    }
    printf("生产者完成任务，线程结束运行\n");

    return 0;
}

//消费者线程函数
unsigned int __stdcall ConsumerThreadFun(PVOID pM)
{
    while (true)
    {
        //等待非空的缓冲区出现
        WaitForSingleObject(g_hSemaphoreBufferFull, INFINITE);

        //互斥的访问缓冲区
        EnterCriticalSection(&g_cs);
        SetConsoleColor(FOREGROUND_GREEN);
        printf("  编号为%d的消费者从缓冲池中第%d个缓冲区取出数据%d\n", GetCurrentThreadId(), g_j, g_Buffer[g_j]);
        SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        if (g_Buffer[g_j] == END_PRODUCE_NUMBER) //结束标志
        {
            LeaveCriticalSection(&g_cs);
            //通知其它消费者有新数据了(结束标志)
            ReleaseSemaphore(g_hSemaphoreBufferFull, 1, NULL);
            break;
        }
        g_j = (g_j + 1) % BUFFER_SIZE;
        LeaveCriticalSection(&g_cs);

        Sleep(50); //some other work to do

        ReleaseSemaphore(g_hSemaphoreBufferEmpty, 1, NULL);
    }
    SetConsoleColor(FOREGROUND_GREEN);
    printf("  编号为%d的消费者收到通知，线程结束运行\n", GetCurrentThreadId());
    SetConsoleColor(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);

    return 0;
}

// ===============================
int main(int argc, char **argv)
{
    printf("  生产者消费者问题   1生产者 2消费者 4缓冲区\n");
    printf(" -- by MoreWindows( http://blog.csdn.net/MoreWindows ) --\n\n");

    InitializeCriticalSection(&g_cs);
    //初始化信号量,一个记录有产品的缓冲区个数,另一个记录空缓冲区个数.
    g_hSemaphoreBufferEmpty = CreateSemaphore(NULL, 4, 4, NULL);
    g_hSemaphoreBufferFull  = CreateSemaphore(NULL, 0, 4, NULL);
    g_i                     = 0;
    g_j                     = 0;
    std::memset(g_Buffer, 0, sizeof(g_Buffer));

    const int THREADNUM = 3;
    HANDLE    hThread[THREADNUM];
    //生产者线程
    hThread[0] = (HANDLE)_beginthreadex(NULL, 0, ProducerThreadFun, NULL, 0, NULL);
    //消费者线程
    hThread[1] = (HANDLE)_beginthreadex(NULL, 0, ConsumerThreadFun, NULL, 0, NULL);
    hThread[2] = (HANDLE)_beginthreadex(NULL, 0, ConsumerThreadFun, NULL, 0, NULL);
    WaitForMultipleObjects(THREADNUM, hThread, TRUE, INFINITE);
    for (int i = 0; i < THREADNUM; i++) CloseHandle(hThread[i]);

    //销毁信号量和关键段
    CloseHandle(g_hSemaphoreBufferEmpty);
    CloseHandle(g_hSemaphoreBufferFull);
    DeleteCriticalSection(&g_cs);

    return 0;
}
