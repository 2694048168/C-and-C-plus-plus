/**
 * @file lock_unlock_main.cpp
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
#include <map>
#include <mutex>
#include <thread>
#include <vector>

std::mutex    g_mutex;
unsigned long g_counter;

void Incrementer()
{
    for (size_t i = 0; i < 100; ++i)
    {
        g_mutex.lock();
        ++g_counter;
        g_mutex.unlock();
    }
}

/**
 * @brief demo for mutex and locks
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::map<int, int> count_map;

    const int num_epochs = 1000;
    for (int i = 0; i < num_epochs; ++i)
    {
        g_counter = 0;

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
        // std::cout << "g_counter: " << g_counter << std::endl;
        std::cout << g_counter << ", ";

        count_map[g_counter]++;
    }
    std::cout << "\n" << std::endl;

    // Assert that we always get 100*100 (we count to 100, 100 times.)
    assert(count_map[100 * 100] == num_epochs);

    return 0;
}