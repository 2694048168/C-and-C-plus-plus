/**
 * @file multithreading_condition.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

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

/* 2. 使用条件变量(condition variables)
-------------------------------------- */
struct Packet
{
    int id;
};

std::condition_variable cond_var;
std::mutex              m;
std::queue<Packet>      packet_queue;

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

        cond_var.notify_one();
    }
}

// 消费者 - 签收包裹
void recipient()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock{m};

        /* 该 lambda 函数必须要返回 true/false */
        cond_var.wait(lock, []() { return !packet_queue.empty(); });

        auto new_packet = packet_queue.front();
        packet_queue.pop();
        std::cout << "Received new packet with id: [" << new_packet.id << "]. [" << packet_queue.size()
                  << "] packets remaining" << std::endl;

        lock.unlock();
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
    // https://zhuanlan.zhihu.com/p/607028487
    /* 1. 周期性检查条件(最简单但是最差的实现方法)
    ---------------------------------------- */
    std::thread producing_threading(parcel_delivery);
    std::thread consuming_threading(recipient);

    producing_threading.join();
    consuming_threading.join();

    return 0;
}
