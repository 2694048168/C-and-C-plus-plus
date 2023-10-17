/**
 * @file 07_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstddef>
#include <functional>
#include <condition_variable>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// [C++线程的使用](https://subingwen.cn/cpp/thread/)
// [命名空间 - this_thread](https://subingwen.cn/cpp/this_thread/)
// [多线程环境下 call_once](https://subingwen.cn/cpp/call_once/)
// [线程同步之互斥锁](https://subingwen.cn/cpp/mutex/)
// [线程同步之条件变量](https://subingwen.cn/cpp/condition/)
// [原子变量和原子操作](https://subingwen.cn/cpp/atomic/)
// [多线程异步操作](https://subingwen.cn/cpp/async/)

// clang++ 07_thread.cpp -std=c++17
// clang++ 07_thread.cpp -std=c++20
// clang++ 07_thread.cpp -std=c++23

// g++ 07_thread.cpp -std=c++20
// g++ 07_thread.cpp -std=c++23

// 可以使用 std::call_once() 来保证函数在多线程环境下只能被调用一次
std::once_flag g_flag;

void do_once(int a, std::string b)
{
    std::cout << "name: " << b << ", age: " << a << "\n";
}

void do_something(int age, std::string name)
{
    static int num = 1;
    std::call_once(g_flag, do_once, 19, "Wei Li");
    std::cout << "do_something() function num = " << num++ << "\n";
}

// CSP 并发模型中的 Channel, condition variable 实现线程同步
class SyncQueue
{
public:
    SyncQueue(int maxSize)
        : m_maxSize(maxSize)
    {
    }

    void put(const int &x)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        // 判断任务队列是不是已经满了
        // while (m_queue.size() == m_maxSize)
        // {
        //     std::cout << "the TASK queue is full, [producer] please wait...\n";
        //     // 阻塞线程
        //     m_notFull.wait(locker);
        // }
        /* 条件变量 condition_variable 类的wait() 一个重载的方法
        可以接受一个条件,这个条件也可以是一个返回值为布尔类型的函数,
        条件变量会先检查判断这个条件是否满足,
        如果满足条件（布尔值为true）,则当前线程重新获得互斥锁的所有权,结束阻塞,继续向下执行;
        如果不满足条件（布尔值为false）,当前线程会释放互斥锁（解锁）同时被阻塞,等待被唤醒。
        ------------------------------------------------- */
        // 根据条件阻塞线程
        m_notFull.wait(locker, [this]() { return m_queue.size() != m_maxSize; });

        // 将任务放入到任务队列中
        m_queue.push_back(x);
        std::cout << x << " was produced by producer\n";
        // 通知消费者去消费
        m_notEmpty.notify_one();
    }

    int take()
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        // while (m_queue.empty())
        // {
        //     std::cout << "the TASK queue is empty, [consumer] please wait...\n";
        //     m_notEmpty.wait(locker);
        // }
        m_notEmpty.wait(locker, [this]() { return !m_queue.empty(); });

        // 从任务队列中取出任务(消费)
        int x = m_queue.front();
        m_queue.pop_front();
        // 通知生产者去生产
        m_notFull.notify_one();
        std::cout << x << " was consumed by consumer\n";

        return x;
    }

    bool empty()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_queue.empty();
    }

    bool full()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_queue.size() == m_maxSize;
    }

    int size()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_queue.size();
    }

private:
    std::list<int>          m_queue;    // 存储队列数据
    std::mutex              m_mutex;    // 互斥锁
    std::condition_variable m_notEmpty; // 不为空的条件变量
    std::condition_variable m_notFull;  // 没有满的条件变量
    const unsigned int      m_maxSize;  // 任务队列的最大任务个数
};

// 原子变量和原子操作
struct Counter
{
    void increment()
    {
        for (int i = 0; i < 10; ++i)
        {
            m_value++;
            std::cout << "increment number: " << m_value << ", theadID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void decrement()
    {
        for (int i = 0; i < 10; ++i)
        {
            m_value--;
            std::cout << "decrement number: " << m_value << ", theadID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // atomic<int> == atomic_int
    std::atomic_int m_value = 0;
};

// ------------------------------
int main(int argc, char **argv)
{
    const unsigned NUM_CORE = std::thread::hardware_concurrency();
    std::cout << "the number of CPU cores is: " << NUM_CORE << "\n";

    // std::vector<std::thread> threads(NUM_CORE);
    std::thread t1(do_something, 20, "su");
    std::thread t2(do_something, 22, "pu");
    std::thread t3(do_something, 23, "ou");
    std::thread t4(do_something, 24, "lu");

    if (t1.joinable())
        t1.join();

    if (t2.joinable())
        t2.join();

    if (t3.joinable())
        t3.join();

    if (t3.joinable())
        t3.join();

    if (t4.joinable())
        t4.join();

    std::cout << "===============================\n";
    SyncQueue taskQ(1);

    auto produce = std::bind(&SyncQueue::put, &taskQ, std::placeholders::_1);
    auto consume = std::bind(&SyncQueue::take, &taskQ);

    std::thread t6[3];
    std::thread t7[3];
    for (int i = 0; i < 3; ++i)
    {
        t6[i] = std::thread(produce, i + 100);
        t7[i] = std::thread(consume);
    }

    for (int i = 0; i < 3; ++i)
    {
        t6[i].join();
        t7[i].join();
    }

    std::cout << "===============================\n";
    Counter c;

    auto increment = std::bind(&Counter::increment, &c);
    auto decrement = std::bind(&Counter::decrement, &c);

    std::thread t8(increment);
    std::thread t9(decrement);

    t8.join();
    t9.join();

    return 0;
}
