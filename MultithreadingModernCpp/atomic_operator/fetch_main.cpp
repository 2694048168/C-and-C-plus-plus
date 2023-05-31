/**
 * @file fetch_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <assert.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::atomic<long long> data_1, data_2;

void do_work_1()
{
    data_1++;
}

void do_work_2()
{
    data_2.fetch_add(1);
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

    for (int i = 0; i < 5; i++)
    {
        threads.push_back(std::thread(do_work_1));
        threads.push_back(std::thread(do_work_2));
    }

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "data_1: " << data_1 << std::endl;
    std::cout << "data_2: " << data_2 << std::endl;
    auto y = data_2 += 7;
    std::cout << "y: " << y << std::endl;

    assert(data_1 == data_2);

    return 0;
}