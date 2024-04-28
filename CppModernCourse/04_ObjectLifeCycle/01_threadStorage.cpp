/**
 * @file 01_threadStorage.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdio>
#include <thread>

/**
 * @brief 线程局部存储期 Thread-Local Storage Duration
 * 并发程序的基本概念之一是线程, 每个程序都有一个或多个线程, 它们可以执行独立的操作;
 * 线程执行的指令序列称为它的线程执行(thread of execution);
 * 程序员在使用多个线程执行时必须采取额外的预防措施, 多个线程能够安全执行的代码称为线程安全代码;
 * 可变全局变量是许多线程安全问题的根源, 可以通过为每个线程提供单独的变量副本来避免这些问题,
 * 这可以通过指定具有线程存储期的对象来实现, 可以通过在 static 或 extern 关键字之外添加 
 * thread_local 关键字来修改具有静态存储期的变量, 使其具有线程局部存储期.
 *  如果只指定了 thread_local, 则隐含 static 声明, 变量的链接方式不变(external).
 * 
 */

static int g_val = 22;

static thread_local int g_val_thread = 22;

void write_globalShared()
{
    ++g_val;
    ++g_val_thread;
}

void read_globalShared()
{
    printf("the global shared value: %d\n", g_val);
    printf("the global thread local value: %d\n", g_val_thread);
}

// ------------------------------------
int main(int argc, const char **argv)
{
    static const size_t cycle = 2;
    for (size_t idx = 0; idx < cycle; ++idx)
    {
        std::thread thread_write{write_globalShared};
        std::thread thread_read{read_globalShared};
        std::thread thread_read_{read_globalShared};

        thread_write.join();
        thread_read.join();
        thread_read_.join();
    }

    return 0;
}
