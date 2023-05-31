/**
 * @file lock_guard_main.cpp
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

std::mutex        g_mutex;
unsigned long int g_counter;

void Incrementer()
{
    for (size_t i = 0; i < 100; ++i)
    {
        std::lock_guard<std::mutex> guard(g_mutex);
        ++g_counter;
    }
}

/**
 * @brief for mutex and lock_guard
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
        if (t.joinable())
            t.join();
    }
    std::cout << "g_counter: " << g_counter << std::endl;

    assert(g_counter == 100 * 100);

    return 0;
}