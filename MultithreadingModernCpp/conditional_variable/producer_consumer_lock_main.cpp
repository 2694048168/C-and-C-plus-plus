/**
 * @file producer_consumer_lock_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>


// Create a function to generate a random value between 0 and 10
auto GenRandomValue = std::bind(std::uniform_int_distribution<>(0, 10), std::default_random_engine());

std::mutex g_mutex;
bool       g_ready = false;
int        g_data  = 0;

/**
 * @brief Uses busy waiting on g_ready to get data from producer
 */
void consumer()
{
    int receivedCount = 0;

    for (int i = 0; i < 100; i++)
    {
        std::unique_lock<std::mutex> ul(g_mutex);
        // Busy waiting
        while (!g_ready)
        {
            ul.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ul.lock();
        }

        std::cout << "got g_data: " << g_data << std::endl;
        g_ready = false;
    }
}

/**
 * @brief Produces data and then sets g_ready to true
 */
void producer()
{
    for (int i = 0; i < 100; i++)
    {
        std::unique_lock<std::mutex> ul(g_mutex);

        // Produce data
        g_data = GenRandomValue();

        // Announce that data is produced
        g_ready = true;

        ul.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ul.lock();
    }
}

/**
 * @brief A demo of producer/consumer using busy waiting on shared memory
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::thread t1(consumer);
    std::thread t2(producer);

    t1.join();
    t2.join();

    return 0;
}