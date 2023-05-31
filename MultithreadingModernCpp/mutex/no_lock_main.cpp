/**
 * @file no_lock_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ no_lock_main.cpp -std=c++17
 * $ g++ no_lock_main.cpp -std=c++17
 * 
 */
#include <assert.h>
#include <stdint.h>

#include <iostream>
#include <map>
#include <thread>
#include <vector>

// global shared memory
unsigned long int g_counter{0};

void Incrementer()
{
    for (size_t i = 0; i < 100; ++i)
    {
        ++g_counter;
    }
}

/**
 * @brief incorrectly using shared memory without mutex and locks
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::map<int, int> count;

    const size_t num_epochs = 1000;
    for (size_t i = 0; i < num_epochs; ++i)
    {
        g_counter = 0;
        std::vector<std::thread> threads;

        for (size_t i = 0; i < 100; ++i)
        {
            threads.push_back(std::thread(Incrementer));
        }

        for (std::thread &t : threads)
        {
            if (t.joinable())
                t.join();
        }

        // std::cout << "the shared memory value: " << g_counter << ", ";
        std::cout << g_counter << ", ";
    }
    std::cout << "\n" << std::endl;

    assert(count[100 * 100] == num_epochs);

    return 0;
}