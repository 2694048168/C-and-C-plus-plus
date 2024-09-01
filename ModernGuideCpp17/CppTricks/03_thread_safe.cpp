/**
 * @file 03_thread_safe.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++中确保线程安全的几种方式
 * @version 0.1
 * @date 2024-09-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>

// 1.使用互斥量（mutex）来保护共享资源
#if 0
std::mutex g_mutex;       // 全局互斥量
int        g_counter = 0; // 共享资源

void incrementCounter()
{
    std::lock_guard<std::mutex> lock(g_mutex); // 上锁
    ++g_counter;
}
#endif

// 2. 使用读写锁（reader-writer lock）来保护共享资源
#if 0
std::shared_mutex g_mutex;       // 全局读写锁
int               g_counter = 0; // 共享资源

void incrementCounter()
{
    std::unique_lock<std::shared_mutex> lock(g_mutex); // 上写锁
    ++g_counter;
}

void readCounter()
{
    std::shared_lock<std::shared_mutex> lock(g_mutex); // 上读锁
    std::cout << "g_counter = " << g_counter << std::endl;
}
#endif

// 3.使用原子操作来保护共享资源
#if 0
std::atomic<int> g_counter{0}; // 原子变量

void incrementCounter()
{
    ++g_counter;
}
#endif

// 4.使用条件变量（condition variable）来协调线程间的协作
#if 0
std::mutex              g_mutex;
std::condition_variable g_cv;
bool                    g_flag = false;

void thread1()
{
    std::unique_lock<std::mutex> lock(g_mutex);
    g_flag = true;
    g_cv.notify_one(); // 通知线程 2
}

void thread2()
{
    std::unique_lock<std::mutex> lock(g_mutex);
    while (!g_flag)
    {
        g_cv.wait(lock); // 等待通知
    }
    std::cout << "thread 2 finished" << std::endl;
}
#endif

// 5.使用线程本地存储（thread-local storage）来保存线程的私有数据
#if 1
thread_local int g_counter = 0; // 线程本地变量

void incrementCounter()
{
    ++g_counter;
    std::cout << std::this_thread::get_id() << ": " << g_counter << std::endl;
}
#endif

// ---------------------------------------
int main(int argc, const char **argv)
{
#if 0
    // 1.使用互斥量（mutex）来保护共享资源
    std::thread t1(incrementCounter);
    std::thread t2(incrementCounter);

    t1.join();
    t2.join();

    std::cout << "g_counter = " << g_counter << std::endl;
#endif

#if 0
    // 2. 使用读写锁（reader-writer lock）来保护共享资源
    std::thread t1(incrementCounter);
    std::thread t2(readCounter);

    t1.join();
    t2.join();
#endif

#if 0
    // 3.使用原子操作来保护共享资源
    std::thread t1(incrementCounter);
    std::thread t2(incrementCounter);
    std::thread t3(incrementCounter);
    std::thread t4(incrementCounter);
    std::thread t5(incrementCounter);
    std::thread t6(incrementCounter);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();

    std::cout << "g_counter = " << g_counter.load() << std::endl;
#endif

#if 0
    // 4.使用条件变量（condition variable）来协调线程间的协作
    std::thread t1(thread1);
    std::thread t2(thread2);

    t1.join();
    t2.join();
#endif

    // 5.使用线程本地存储（thread-local storage）来保存线程的私有数据
    std::thread t1(incrementCounter);
    std::thread t2(incrementCounter);

    t1.join();
    t2.join();
    // 每个线程的 g_counter 都是线程本地的,互不干扰,使用线程本地存储来保存线程的私有数据

    return 0;
}
