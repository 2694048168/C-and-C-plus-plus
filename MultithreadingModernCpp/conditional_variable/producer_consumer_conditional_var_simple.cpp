/**
 * @file producer_consumer_conditional_var_simple.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 * ! the thread ID is different by g++ and clang++(MSVC)
 * 
 */

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex              mtx;
std::condition_variable cv;
bool                    ready = false;

void print_id(int id)
{
    std::unique_lock<std::mutex> lck(mtx);
    while (!ready)
    {
        cv.wait(lck);
    }
    // ...
    std::cout << "thread " << id << '\n';
}

void go()
{
    std::unique_lock<std::mutex> lck(mtx);
    ready = true;
    cv.notify_all();
}

/**
 * @brief demo of producer/Consumer using C++ conditional variable
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i = 0; i < 10; ++i)
    {
        threads[i] = std::thread(print_id, i);
        std::cout << "the thread ID: " << threads[i].get_id() << std::endl;
    }

    std::cout << "\n10 threads ready to race...\n";
    go(); // go!

    for (auto &th : threads)
    {
        th.join();
    }

    return 0;
}