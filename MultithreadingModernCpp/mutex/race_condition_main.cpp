/**
 * @file race_condition_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ race_condition_main.cpp -std=c++17
 * $ g++ race_condition_main.cpp -std=c++17
 * 
 */

#include <iostream>
#include <thread>

// global shared memory
unsigned long int g_var;

/**
 * @brief incorrectly using shared memory without mutex and locks
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const size_t num_epochs = 100;
    for (size_t i = 0; i < num_epochs; i++)
    {
        auto t1 = std::thread([]() { g_var = 1; });
        auto t2 = std::thread([]() { g_var = 2; });

        if (t1.joinable())
            t1.join();
        if (t2.joinable())
            t2.join();

        std::cout << "the shared memeory value: " << g_var << ", ";
    }
    std::cout << "\n" << std::endl;

    return 0;
}
