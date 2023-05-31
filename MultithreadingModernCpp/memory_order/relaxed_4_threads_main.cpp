/**
 * @file relaxed_4_threads_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

std::atomic<bool> x = {false};
std::atomic<bool> y = {false};
std::atomic<int>  z = {0};

void write_x()
{
    x.store(true, std::memory_order_relaxed);
}

void write_y()
{
    y.store(true, std::memory_order_relaxed);
}

void read_x_then_y()
{
    while (!x.load(std::memory_order_relaxed))
        ;
    if (y.load(std::memory_order_relaxed))
    {
        ++z;
    }
}

void read_y_then_x()
{
    while (!y.load(std::memory_order_relaxed))
        ;
    if (x.load(std::memory_order_relaxed))
    {
        ++z;
    }
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<std::thread> threads;

    threads.push_back(std::thread(write_x));
    threads.push_back(std::thread(write_y));
    threads.push_back(std::thread(read_x_then_y));
    threads.push_back(std::thread(read_y_then_x));

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    assert(z.load() != 0); // Can happen!
}