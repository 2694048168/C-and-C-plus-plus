/**
 * @file multithreading.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

// A dummy function
void foo(int Z)
{
    for (int i = 0; i < Z; ++i)
    {
        std::cout << "[function]Thread using function"
                     " pointer as callable\n";
    }
}

// A callable object
class ThreadObj
{
public:
    void operator()(int x)
    {
        for (int i = 0; i < x; ++i)
            std::cout << "[class]Thread using function"
                         " object as callable\n";
    }
};

void print(const int n, const std::string &str)
{
    std::string msg = std::to_string(n) + " : " + str;
    std::cout << msg << std::endl;
}

/**
 * @brief 多线程的两个主要用途
 * 1. 保护共享数据(Protecting shared data from concurrent access)
 * 2. 同步并行操作(Synchronizing concurrent operations)
 * 
 * 同步并行操作的三种方法
 * 1. 周期性检查条件(最简单但是最差的实现方法)
 * 2. 使用条件变量(condition variables)
 * 3. 使用futures和promises(理解起来比较复杂, 但有些场景可以很方便)
 * 
 */

/* 1. 周期性检查条件(最简单但是最差的实现方法)
---------------------------------------- */
struct Packet
{
    int id;
};

std::mutex         m;
std::queue<Packet> packet_queue;

// 生产者 - 生成快递包裹
void parcel_delivery()
{
    int packet_id = 0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        Packet new_packet{packet_id};
        {
            std::lock_guard<std::mutex> l{m};

            std::cout << "Work is done and packet id [" << packet_id << "] is ready" << std::endl;

            packet_queue.push(new_packet);
        }
        ++packet_id;
    }
}

// 消费者 - 签收包裹
void recipient()
{
    while (true)
    {
        {
            std::lock_guard<std::mutex> l{m};

            if (!packet_queue.empty())
            {
                auto new_packet = packet_queue.front();
                packet_queue.pop();

                std::cout << "Received new packet with id: [" << new_packet.id << "]. [" << packet_queue.size()
                          << "] packets remaining" << std::endl;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/**
 * @brief Multithreading is a feature that allows concurrent execution of
 * two or more parts of a program for maximum utilization of the CPU in Modern C++.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    /* ======================================= */
    std::cout << "Threads 1 and 2 and 3 "
                 "operating independently"
              << std::endl;

    // This thread is launched by using function pointer as callable
    std::thread thread1(foo, 3);

    // This thread is launched by using function object as callable
    std::thread thread2(ThreadObj(), 3);

    // Define a Lambda Expression
    auto f = [](int x)
    {
        for (int i = 0; i < x; ++i)
            std::cout << "[lambda]Thread using lambda"
                         " expression as callable\n";
    };

    // This thread is launched by using lambda expression as callable
    std::thread thread3(f, 3);

    // Wait for the threads to finish via join()
    thread1.join();
    thread2.join();
    thread3.join();

    /* ======================================= */
    // https://www.educative.io/blog/modern-multithreading-and-concurrency-in-cpp
    std::vector<std::string> s = {"Educative.blog", "Educative", "courses", "are great"};

    std::vector<std::thread> threads;
    for (int i = 0; i < s.size(); i++)
    {
        threads.push_back(std::thread(print, i, s[i]));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    /* ======================================= */
    // https://zhuanlan.zhihu.com/p/607028487
    /* 1. 周期性检查条件(最简单但是最差的实现方法)
    ---------------------------------------- */
    std::thread producing_threading(parcel_delivery);
    std::thread consuming_threading(recipient);

    producing_threading.join();
    consuming_threading.join();

    return 0;
}
