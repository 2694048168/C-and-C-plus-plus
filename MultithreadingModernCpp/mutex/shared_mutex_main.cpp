/**
 * @file shared_mutex_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ shared_mutex_main.cpp -std=c++17
 * $ g++ shared_mutex_main.cpp -std=c++17
 * 
 * ! the result of clang++(same as MSVC) and g++ are different
 * 
 */

#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

// the global shared variables
std::shared_mutex g_shared_mutex;
unsigned long int g_counter;

void Incrementer()
{
    for (size_t i = 0; i < 100; ++i)
    {
        std::unique_lock<std::shared_mutex> ul(g_shared_mutex);
        ++g_counter;
    }
}

void ImJustAReader()
{
    for (size_t i = 0; i < 100; ++i)
    {
        std::shared_lock<std::shared_mutex> sl(g_shared_mutex);
        std::cout << "g_counter: " << g_counter << "\n";
    }
}

/**
 * @brief shared_mutex and shared_lock
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<std::thread> threads;

    for (size_t i = 0; i < 100; ++i)
    {
        threads.push_back(std::thread(Incrementer));
        threads.push_back(std::thread(ImJustAReader));
    }

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "\nthe shared memory value: " << g_counter << std::endl;

    return 0;
}