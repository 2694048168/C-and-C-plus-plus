/**
 * @file unique_lock_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <assert.h>

#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex    g_mutex;
unsigned long g_counter;

void Incrementer()
{
    for (size_t i = 0; i < 100; ++i)
    {
        std::unique_lock<std::mutex> ul(g_mutex);
        ++g_counter;

        ul.unlock();
        std::cout << "Doing something non-critical..." << std::endl;
        ul.lock();
    }
}

/**
 * @brief example for unique_lock: similar to lock_guard, 
 * but it can be lock and unlock multiple times.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<std::thread> threads;

    for (int i = 0; i < 100; ++i)
    {
        threads.push_back(std::thread(Incrementer));
    }

    for (std::thread &t : threads)
    {
        t.join();
    }
    std::cout << "\ng_counter: " << g_counter << std::endl;

    assert(g_counter == 100 * 100);

    return 0;
}